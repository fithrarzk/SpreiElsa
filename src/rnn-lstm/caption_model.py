import numpy as np
from layers import EmbeddingLayer, DenseLayer
from rnn    import SimpleRNNCell, SimpleRNNDecoder
from lstm   import LSTMCell, LSTMDecoder


class CaptionModel:
    def __init__(self, decoder_type: str = "lstm", injection_method: str = "pre"):
        if decoder_type.lower() not in ("rnn", "lstm"):
            raise ValueError("decoder_type must be 'rnn' or 'lstm'")
        if injection_method not in ("pre", "init"):
            raise ValueError("injection_method must be 'pre' or 'init'")
        self.decoder_type      = decoder_type.lower()
        self.injection_method  = injection_method
        self.embedding         = EmbeddingLayer()
        self.image_projection  = DenseLayer()
        self.decoder           = None
        self.output_layer      = DenseLayer(activation="softmax")
        self.word2idx: dict    = None
        self.idx2word: dict    = None
        self.n_layers: int     = None

    def load_from_keras(self, keras_model, word2idx: dict, idx2word: dict) -> None:
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.embedding.load_weights(keras_model.get_layer("embedding"))
        self.image_projection.load_weights(keras_model.get_layer("image_projection"))

        if self.decoder_type == "rnn":
            prefix, CellCls, DecCls = "simple_rnn", SimpleRNNCell, SimpleRNNDecoder
        else:
            prefix, CellCls, DecCls = "lstm", LSTMCell, LSTMDecoder

        cells, i = [], 0
        while True:
            try:
                layer = keras_model.get_layer(f"{prefix}_{i}")
            except ValueError:
                break
            cell = CellCls()
            cell.load_weights(layer)
            cells.append(cell)
            i += 1

        if not cells:
            raise ValueError(f"No layers named '{prefix}_0', '{prefix}_1', ... found in model.")
        self.n_layers = len(cells)
        self.decoder  = DecCls(cells)
        self.output_layer.load_weights(keras_model.get_layer("output_dense"))

    def load_weights_manual(
        self,
        embedding_W, image_proj_W, image_proj_b,
        recurrent_weights, output_W, output_b,
        word2idx, idx2word,
    ) -> None:
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.embedding.set_weights(embedding_W)
        self.image_projection.set_weights(image_proj_W, image_proj_b)

        CellCls = SimpleRNNCell if self.decoder_type == "rnn" else LSTMCell
        DecCls  = SimpleRNNDecoder if self.decoder_type == "rnn" else LSTMDecoder
        cells = []
        for W_x, W_h, b in recurrent_weights:
            cell = CellCls()
            cell.set_weights(W_x, W_h, b)
            cells.append(cell)
        self.decoder  = DecCls(cells)
        self.n_layers = len(cells)
        self.output_layer.set_weights(output_W, output_b)


    def _init_states(self, batch_size=None):
        """Return (h, c) where c is None for RNN; shapes are (n,) or (batch,n)."""
        hs = self.decoder.cells[0].hidden_size
        if batch_size is None:
            h = [np.zeros(hs, dtype=np.float32) for _ in range(self.n_layers)]
            c = [np.zeros(hs, dtype=np.float32) for _ in range(self.n_layers)] \
                if self.decoder_type == "lstm" else None
        else:
            h = [np.zeros((batch_size, hs), dtype=np.float32) for _ in range(self.n_layers)]
            c = [np.zeros((batch_size, hs), dtype=np.float32) for _ in range(self.n_layers)] \
                if self.decoder_type == "lstm" else None
        return h, c

    def _step(self, x, h, c):
        """One decoder step; returns (out, h, c) where c is None for RNN."""
        if self.decoder_type == "rnn":
            out, h = self.decoder.step(x, h)
            return out, h, None
        out, h, c = self.decoder.step(x, h, c)
        return out, h, c

    def _predict(self, out, x_img):
        """Apply output layer; concatenate image for init-inject."""
        if self.injection_method == "init":
            out = np.concatenate([out, x_img], axis=-1)
        return self.output_layer.forward(out)


    def generate_caption(self, image_feature: np.ndarray, max_len: int = 30) -> str:
        assert self.decoder is not None, "Call load_from_keras() first."
        start = self.word2idx["<start>"]
        end   = self.word2idx["<end>"]
        pad   = self.word2idx["<pad>"]

        h, c   = self._init_states()
        x_img  = self.image_projection.forward(image_feature.astype(np.float32))
        generated = []

        if self.injection_method == "pre":
            out, h, c = self._step(x_img, h, c)
            token = int(np.argmax(self._predict(out, x_img)))
            if token in (end, pad):
                return ""
            generated.append(self.idx2word.get(token, "<unk>"))
            to_feed      = start
            prev_pred    = token

            for _ in range(max_len - 1):
                x         = self.embedding.forward(to_feed)
                out, h, c = self._step(x, h, c)
                token     = int(np.argmax(self._predict(out, x_img)))
                if token in (end, pad):
                    break
                generated.append(self.idx2word.get(token, "<unk>"))
                to_feed   = prev_pred
                prev_pred = token

        else:  # init-inject
            token = start
            for _ in range(max_len):
                x         = self.embedding.forward(token)
                out, h, c = self._step(x, h, c)
                token     = int(np.argmax(self._predict(out, x_img)))
                if token in (end, pad):
                    break
                generated.append(self.idx2word.get(token, "<unk>"))

        return " ".join(generated)

    def generate_caption_beam(
        self,
        image_feature: np.ndarray,
        max_len: int = 30,
        beam_size: int = 3,
        length_penalty: float = 0.7,
    ) -> str:
        assert self.decoder is not None, "Call load_from_keras() first."
        if beam_size < 1:
            raise ValueError("beam_size must be >= 1")

        start = self.word2idx["<start>"]
        end   = self.word2idx["<end>"]
        pad   = self.word2idx["<pad>"]

        def score_norm(score, length):
            return score / (length ** length_penalty) if length_penalty > 0 else score

        def copy_state(h, c):
            hc = [np.array(hi, copy=True) for hi in h]
            cc = [np.array(ci, copy=True) for ci in c] if c is not None else None
            return hc, cc

        h0, c0 = self._init_states()
        x_img   = self.image_projection.forward(image_feature.astype(np.float32))

        if self.injection_method == "pre":
            out0, h0, c0 = self._step(x_img, h0, c0)
            probs0   = self._predict(out0, x_img)
            top_idx0 = np.argsort(probs0)[-beam_size:][::-1]
            beams = []
            for idx in top_idx0:
                hc, cc = copy_state(h0, c0)
                tok = int(idx)
                beams.append((
                    [tok],
                    float(np.log(max(float(probs0[tok]), 1e-9))),
                    hc, cc,
                    start,   # to_feed (next: <start>)
                    tok,     # prev_pred (after <start>: feed tok)
                ))
        else:
            h0c, c0c = copy_state(h0, c0)
            beams = [([start], 0.0, h0c, c0c, start, start)]

        for _ in range(max_len):
            candidates = []
            for tokens, score, h, c, to_feed, prev_pred in beams:
                last = tokens[-1]
                if last in (end, pad):
                    candidates.append((tokens, score, h, c, to_feed, prev_pred))
                    continue

                if self.injection_method == "pre":
                    feed_tok = to_feed
                else:
                    feed_tok = last

                x         = self.embedding.forward(feed_tok)
                out, nh, nc = self._step(x, h, c)
                probs     = self._predict(out, x_img)
                top_idx   = np.argsort(probs)[-beam_size:][::-1]

                next_to_feed = prev_pred if self.injection_method == "pre" else None
                for idx in top_idx:
                    prob = float(probs[int(idx)])
                    hc, cc = copy_state(nh, nc)
                    new_prev = int(idx) if self.injection_method == "pre" else None
                    candidates.append((
                        tokens + [int(idx)],
                        score + float(np.log(max(prob, 1e-9))),
                        hc, cc,
                        next_to_feed,
                        new_prev,
                    ))

            candidates.sort(
                key=lambda item: score_norm(item[1], len(item[0])), reverse=True
            )
            beams = candidates[:beam_size]
            if all(t[-1] in (end, pad) for t, *_ in beams):
                break

        best_tokens = max(beams, key=lambda item: score_norm(item[1], len(item[0])))[0]
        words = [
            self.idx2word.get(t, "<unk>")
            for t in best_tokens
            if t not in (start, pad, end)
        ]
        return " ".join(words)

    def forward_sequence(self, image_feature: np.ndarray, caption_tokens: np.ndarray) -> np.ndarray:
        assert self.decoder is not None
        x_img   = self.image_projection.forward(image_feature.astype(np.float32))
        cap_emb = self.embedding.forward(np.array(caption_tokens))

        if self.injection_method == "pre":
            full_seq = np.concatenate([x_img[np.newaxis], cap_emb], axis=0)
        else:
            full_seq = cap_emb

        if self.decoder_type == "rnn":
            outputs, _ = self.decoder.forward(full_seq)
        else:
            outputs, _, _ = self.decoder.forward(full_seq)

        if self.injection_method == "init":
            outputs = np.concatenate(
                [outputs, np.repeat(x_img[np.newaxis], outputs.shape[0], axis=0)], axis=-1
            )
        return np.stack([self.output_layer.forward(outputs[t]) for t in range(len(outputs))], axis=0)

    def forward_sequence_batch(
        self, image_features: np.ndarray, caption_tokens: np.ndarray
    ) -> np.ndarray:
        assert self.decoder is not None
        x_img   = self.image_projection.forward(image_features.astype(np.float32))
        cap_emb = self.embedding.forward(np.array(caption_tokens))

        if self.injection_method == "pre":
            full_seq = np.concatenate([x_img[:, np.newaxis, :], cap_emb], axis=1)
        else:
            full_seq = cap_emb

        if self.decoder_type == "rnn":
            outputs, _ = self.decoder.forward(full_seq)
        else:
            outputs, _, _ = self.decoder.forward(full_seq)

        if self.injection_method == "init":
            outputs = np.concatenate(
                [outputs, x_img[:, np.newaxis, :].repeat(outputs.shape[1], axis=1)], axis=-1
            )
        flat        = outputs.reshape((-1, outputs.shape[-1]))
        logits_flat = self.output_layer.forward(flat)
        return logits_flat.reshape((outputs.shape[0], outputs.shape[1], -1))

    def generate_captions_batch(
        self, image_features: np.ndarray, max_len: int = 30, batch_size: int = 32
    ) -> list:
        assert self.decoder is not None, "Call load_from_keras() first."

        captions = []
        for start_row in range(0, len(image_features), batch_size):
            feature_batch = np.asarray(
                image_features[start_row : start_row + batch_size], dtype=np.float32
            )
            if feature_batch.size == 0:
                continue

            pad_idx   = self.word2idx["<pad>"]
            start_idx = self.word2idx["<start>"]
            end_idx   = self.word2idx["<end>"]
            batch     = feature_batch.shape[0]
            hs        = self.decoder.cells[0].hidden_size

            x_img = self.image_projection.forward(feature_batch)
            generated = [[] for _ in range(batch)]
            finished  = np.zeros(batch, dtype=bool)
            h = [np.zeros((batch, hs), dtype=np.float32) for _ in range(self.n_layers)]
            c = (
                [np.zeros((batch, hs), dtype=np.float32) for _ in range(self.n_layers)]
                if self.decoder_type == "lstm"
                else None
            )

            def step(x):
                nonlocal h, c
                if self.decoder_type == "rnn":
                    out, h = self.decoder.step(x, h)
                    return out
                out, h, c = self.decoder.step(x, h, c)
                return out

            def predict(out):
                if self.injection_method == "init":
                    out = np.concatenate([out, x_img], axis=-1)
                return self.output_layer.forward(out)

            if self.injection_method == "pre":
                out = step(x_img)
                tokens = np.argmax(predict(out), axis=-1).astype(np.int32)
                for i, tok in enumerate(tokens):
                    if int(tok) in (end_idx, pad_idx):
                        finished[i] = True
                    else:
                        generated[i].append(self.idx2word.get(int(tok), "<unk>"))

                to_feed = np.full(batch, start_idx, dtype=np.int32)
                prev_pred = tokens.copy()
                for _ in range(max_len - 1):
                    out = step(self.embedding.forward(to_feed))
                    new_tok = np.argmax(predict(out), axis=-1).astype(np.int32)
                    for i, tok in enumerate(new_tok):
                        if not finished[i]:
                            if int(tok) in (end_idx, pad_idx):
                                finished[i] = True
                            else:
                                generated[i].append(self.idx2word.get(int(tok), "<unk>"))
                    to_feed = prev_pred
                    prev_pred = new_tok
                    if finished.all():
                        break
            else:
                to_feed = np.full(batch, start_idx, dtype=np.int32)
                for _ in range(max_len):
                    out = step(self.embedding.forward(to_feed))
                    to_feed = np.argmax(predict(out), axis=-1).astype(np.int32)
                    for i, tok in enumerate(to_feed):
                        if not finished[i]:
                            if int(tok) in (end_idx, pad_idx):
                                finished[i] = True
                            else:
                                generated[i].append(self.idx2word.get(int(tok), "<unk>"))
                    if finished.all():
                        break

            captions.extend(" ".join(words) for words in generated)

        return captions
