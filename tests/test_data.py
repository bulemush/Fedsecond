from fedproxy.data.collator import ResponseOnlyCollator
from fedproxy.data.partition import iid_partition
from fedproxy.data.tasks import TaskExample


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens):
        prefix = [1] if add_special_tokens else []
        return prefix + [3 + (ord(char) % 10) for char in text]


def test_iid_partition_is_disjoint_and_complete():
    partition = iid_partition(list(range(20)), 4, 42)
    flat = [item for values in partition.values() for item in values]
    assert sorted(flat) == list(range(20))
    assert len(flat) == len(set(flat))


def test_response_only_mask_supervises_target_and_eos():
    collator = ResponseOnlyCollator(Tokenizer(), max_input_length=4, max_target_length=4)
    batch = collator([TaskExample("1", "synthetic", "abcdef", "xy")])
    labels = batch["labels"][0].tolist()
    assert labels[:4] == [-100] * 4
    assert labels[-1] == 2
    assert batch["attention_mask"].sum().item() == len(labels)

