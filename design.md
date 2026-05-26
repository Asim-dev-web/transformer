# Architecture Specification: Byte-Pair Encoding (BPE) Tokenizer

## 1. Module Overview

The goal of this module is to implement a high-performance Byte-Pair Encoding (BPE) tokenizer from scratch using **Doubly Linked Lists** and a **Min-Heap (Priority Queue)**. This specific design avoids global string scanning during the encoding loop, optimizing runtime complexity from $O(N^2)$ to $O(M \log K)$ on local CPU hardware.

The system is strictly separated into two decoupled pipelines:

* **The Training Engine:** Discovers optimal statistical byte merges sequentially from a text corpus to build an implicit rulebook.
* **The Inference Engine (Encoding/Decoding):** Applies the frozen rulebook to novel text via localized pointer manipulation in RAM.

---

## 2. Global Core Data Structures

To achieve maximum memory performance and $O(1)$ constant-time lookups during active tokenization, the system utilizes three primary data configurations in RAM:

### A. The Doubly Linked List Node (`Node`)

Represents the text sequence in memory as a graph of mutable pointers.

* `value` (int): The current token ID or raw byte integer value.
* `prev` (`Node` pointer / `None`): Reference to the immediate left neighbor.
* `next` (`Node` pointer / `None`): Reference to the immediate right neighbor.

### B. The Encoder Lookup Map (`self.encoder_rules`)

* **Type:** `dict[tuple[int, int], tuple[int, int]]`
* **Structure:** `{(Left_Token_ID, Right_Token_ID): (Priority_Rank, New_Token_ID)}`
* **Purpose:** Allows the execution loop to instantly check if an adjacent pair of nodes forms a valid merge rule, retrieving its priority ranking and target ID in constant time.

### C. The Decoder Lookup Map (`self.decoder_rules`)

* **Type:** `dict[int, bytes]`
* **Structure:** `{Token_ID: raw_bytes_sequence}`
* **Purpose:** Maps final integer predictions from the transformer language model head back into raw, unformatted computer memory buffers.

---

## 3. Data Pipeline & Execution Flow

### Phase 1: Storage Format (On Disk)

To maintain portability and adhere to standard serialization rules, the rulebook is saved as a clean JSON configuration file.

```json
{
  "vocab_size": 1000,
  "merge_rules": [
    "116 104",
    "97 110",
    "256 101"
  ]
}

```

* **Implicit Priority Rank:** The position of the string inside the `merge_rules` array (`index + 1`) automatically dictates its rank.
* **Implicit Production Target:** The target ID created by the merge is mathematically computed as `256 + index`, safely preventing collisions with the base UTF-8 alphabet space ($0 \rightarrow 255$).

---

### Phase 2: Runtime Reconstruction (Bootup Loop)

Upon initialization or calling `load_from_json()`, the tokenizer reads the flat list of strings and dynamically constructs both runtime dictionaries in memory:

```text
1. Initialize self.decoder_rules with base bytes 0 to 255.
2. Initialize empty self.encoder_rules map.
3. FOR index, rule_string IN enumerate(json_data["merge_rules"]):
    │
    ├── Split rule_string by space ──► left_id, right_id
    ├── Compute: rank = index + 1
    ├── Compute: new_token_id = 256 + index
    │
    ├── Populate Encoder Map:
    │    └── self.encoder_rules[(left_id, right_id)] = (rank, new_token_id)
    │
    └── Populate Decoder Map (Glue Bytes):
         └── self.decoder_rules[new_token_id] = self.decoder_rules[left_id] + self.decoder_rules[right_id]

```

---

### Phase 3: The Encoding Loop (Inference Data Flow)

When new string text is passed to the tokenizer, it is processed via a localized repair pipeline:

1. **Graph Initialization:** Convert the string into its raw UTF-8 byte integers ($0 \rightarrow 255$). Instantiate a `Node` object for every byte and link them sequentially to form the initial Doubly Linked List.
2. **Initial Pass:** Traverse the linked list linearly once. For every adjacent pair, check `self.encoder_rules`. If a match is found, push a tuple of `(Priority_Rank, Left_Node_Object, Right_Node_Object)` into a Python `heapq` Min-Heap.
3. **The Pop & Validate Loop:**
* Pop the highest priority item (lowest rank number) from the heap.
* **Validation Check:** Verify if `Left_Node.next == Right_Node`. If false, the pair is stale (already eaten by an earlier merge) $\rightarrow$ discard immediately.


4. **Execution of Merge:**
* Instantiate `new_node = Node(new_token_id)`.
* Set `new_node.prev = Left_Node.prev` and `new_node.next = Right_Node.next`.
* **Edge Case Handling:**
* If `Left_Node.prev` is not `None`, set `Left_Node.prev.next = new_node`. Else, update `head = new_node`.
* If `Right_Node.next` is not `None`, set `Right_Node.next.prev = new_node`. Else, update `tail = new_node`.




5. **Local Repair Step:**
* Check the immediate new left adjacent pair: `(new_node.prev, new_node)`. If valid in the rulebook, push it to the heap.
* Check the immediate new right adjacent pair: `(new_node, new_node.next)`. If valid in the rulebook, push it to the heap.


6. **Termination:** Repeat steps 3–5 until the Min-Heap is completely empty. Traverse the remaining active nodes linearly to output the final sequence of compressed integer token IDs.

---

## 4. Public API Interface Contract

```python
class Tokenizer:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.encoder_rules = {}
        self.decoder_rules = {i: bytes([i]) for i in range(256)}
        self.merge_rules_list = []

    def train(self, corpus_path: str) -> None:
        pass

    def encode(self, text: str) -> list[int]:
        pass

    def decode(self, tokens: list[int]) -> str:
        pass

    def save_to_json(self, file_path: str) -> None:
        pass

    def load_from_json(self, file_path: str) -> None:
        pass

```