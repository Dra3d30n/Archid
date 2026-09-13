import numpy as np
import euclid as Euclid
import archid as Archid


# ============================================================
# Positional Encoding Layer
# ============================================================

class PositionalEncoding(Euclid.layers.Layer):

    def __init__(
        self,
        seq_len,
        d_model,
    ):
        super().__init__()

        position = np.arange(
            seq_len
        )[:, None]

        div_term = np.exp(
            np.arange(
                0,
                d_model,
                2,
            )
            * -(np.log(10000.0) / d_model)
        )

        pe = np.zeros(
            (seq_len, d_model)
        )

        pe[:, 0::2] = np.sin(
            position * div_term
        )

        pe[:, 1::2] = np.cos(
            position * div_term
        )

        self.pe = Euclid.Tensor(
            pe
        )

    def forward(self, x):

        return x + self.pe


# ============================================================
# Configuration
# ============================================================

Euclid.xp.random.seed(7)

VOCAB_SIZE = 4
SEQ_LEN = 8

D_MODEL = 64
NUM_HEADS = 4
NUM_BLOCKS = 3

BATCH_SIZE = 256

EPOCHS = 1000

LEARNING_RATE = 0.001


# ============================================================
# Generate ALL possible sequences
# ============================================================

NUM_SAMPLES = VOCAB_SIZE ** SEQ_LEN

print(
    "Generating dataset..."
)

print(
    "Vocabulary:",
    VOCAB_SIZE
)

print(
    "Sequence length:",
    SEQ_LEN
)

print(
    "Total sequences:",
    NUM_SAMPLES
)


# ============================================================
# Generate every possible sequence
# ============================================================

tokens_np = np.array(
    np.unravel_index(
        np.arange(NUM_SAMPLES),
        (VOCAB_SIZE,) * SEQ_LEN,
    )
).T


# ============================================================
# Reverse sequence
# ============================================================

targets_np = tokens_np[:, ::-1]


# ============================================================
# Move to Euclid backend
# ============================================================

tokens = Euclid.xp.asarray(
    tokens_np
)

targets = Euclid.xp.asarray(
    targets_np
)


# ============================================================
# One-hot encode
# ============================================================

x_data = Euclid.xp.eye(
    VOCAB_SIZE
)[tokens]

target_data = Euclid.xp.eye(
    VOCAB_SIZE
)[targets]


print(
    "Input shape:",
    x_data.shape
)

print(
    "Target shape:",
    target_data.shape
)


# ============================================================
# Build Transformer Sequentially
# ============================================================

layers = []


# ------------------------------------------------------------
# Token embedding
# ------------------------------------------------------------

layers.append(
    Euclid.layers.Dense(
        VOCAB_SIZE,
        D_MODEL,
    )
)


# ------------------------------------------------------------
# Positional encoding
# ------------------------------------------------------------

layers.append(
    PositionalEncoding(
        SEQ_LEN,
        D_MODEL,
    )
)


# ------------------------------------------------------------
# Transformer blocks
# ------------------------------------------------------------

for _ in range(NUM_BLOCKS):

    layers.append(
        Euclid.layers.TransformerBlock(
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            causal=True,
        )
    )


# ------------------------------------------------------------
# Final normalization
# ------------------------------------------------------------

layers.append(
    Euclid.layers.LayerNorm(
        D_MODEL
    )
)


# ------------------------------------------------------------
# Language model head
# ------------------------------------------------------------

layers.append(
    Euclid.layers.Dense(
        D_MODEL,
        VOCAB_SIZE,
    )
)


# ============================================================
# Sequential Model
# ============================================================

model = Euclid.layers.Sequential(
    layers
)


# ============================================================
# Parameters
# ============================================================

parameters = model.parameters()


print(
    "Parameter tensors:",
    len(parameters),
)


parameter_count = sum(
    np.prod(
        parameter.data.shape
    )
    for parameter in parameters
)

print(
    "Total parameters:",
    parameter_count
)


# ============================================================
# Optimizer
# ============================================================

optimizer = Euclid.optimizers.Adam(
    learning_rate=LEARNING_RATE,
)

optimizer.set_parameters(
    parameters
)


# ============================================================
# Loss
# ============================================================

loss_f = Euclid.losses.CrossEntropy()


# ============================================================
# Training
# ============================================================

print(
    "\nTraining...\n"
)


for epoch in range(
    EPOCHS
):

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    permutation = (
        Euclid.xp.random.permutation(
            NUM_SAMPLES
        )
    )

    total_loss = 0.0

    total_correct = 0

    total_tokens = 0


    # --------------------------------------------------------
    # Mini-batches
    # --------------------------------------------------------

    for start in range(
        0,
        NUM_SAMPLES,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            NUM_SAMPLES,
        )

        indices = permutation[
            start:end
        ]


        # ----------------------------------------------------
        # Batch
        # ----------------------------------------------------

        batch_x = x_data[
            indices
        ]

        batch_targets = target_data[
            indices
        ]


        # ----------------------------------------------------
        # Tensor
        # ----------------------------------------------------

        x = Euclid.Tensor(
            batch_x,
            requires_grad=True,
        )

        targets_tensor = Euclid.Tensor(
            batch_targets.reshape(
                -1,
                VOCAB_SIZE,
            )
        )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits = model(x)


        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        logits_flat = logits.reshape(
            -1,
            VOCAB_SIZE,
        )


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = loss_f.forward(
            logits_flat,
            targets_tensor,
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        predictions = (
            Euclid.xp.argmax(
                logits.data,
                axis=-1,
            )
        )

        expected = targets[
            indices
        ]


        total_correct += int(
            Euclid.to_numpy(
                Euclid.xp.sum(
                    predictions == expected
                )
            )
        )


        total_tokens += (
            (end - start)
            * SEQ_LEN
        )


        total_loss += (
            float(loss.data)
            * (end - start)
        )


    # ========================================================
    # Epoch statistics
    # ========================================================

    average_loss = (
        total_loss
        / NUM_SAMPLES
    )

    accuracy = (
        total_correct
        / total_tokens
    )


    if epoch % 10 == 0:

        print(
            f"Epoch {epoch:4d} | "
            f"Loss: {average_loss:.6f} | "
            f"Accuracy: "
            f"{accuracy * 100:.2f}%"
        )


# ============================================================
# Test ALL sequences
# ============================================================

print(
    "\n========================================"
)

print(
    "Testing ALL sequences"
)

print(
    "========================================\n"
)


x_test = Euclid.Tensor(
    x_data
)


logits = model(
    x_test
)


predictions = (
    Euclid.xp.argmax(
        logits.data,
        axis=-1,
    )
)

expected = targets


# ============================================================
# Token accuracy
# ============================================================

accuracy = Euclid.xp.mean(
    predictions == expected
)


print(
    "Token accuracy:",
    f"{float(accuracy) * 100:.2f}%"
)


# ============================================================
# Completely correct sequences
# ============================================================

sequence_correct = (
    Euclid.xp.all(
        predictions == expected,
        axis=1,
    )
)


num_correct = int(
    Euclid.to_numpy(
        sequence_correct
    ).sum()
)


print(
    "Completely correct:",
    f"{num_correct}/{NUM_SAMPLES}"
)


# ============================================================
# Show examples
# ============================================================

print(
    "\nExamples:\n"
)


example_indices = [
    0,
    1,
    2,
    3,
    10,
    100,
    1000,
    10000,
    30000,
    50000,
    65535,
]


tokens_print = (
    Euclid.to_numpy(
        tokens
    )
)

expected_print = (
    Euclid.to_numpy(
        expected
    )
)

predictions_print = (
    Euclid.to_numpy(
        predictions
    )
)


for i in example_indices:

    print(
        "Input:    ",
        tokens_print[i]
    )

    print(
        "Expected: ",
        expected_print[i]
    )

    print(
        "Predicted:",
        predictions_print[i]
    )

    print()