"""Loading a Qwen3.5 tokenizer with the pre-tokenizer transformers uses."""

from tokenizers import Regex, Tokenizer, models, pre_tokenizers

from mlx_decision.backbones.qwen3_5.tokenizer import load_tokenizer

# The pattern the releases' tokenizer.json files carry: no \p{M}.
OLD_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""  # noqa: E501
HINDI = "हिंदी भाषा"


def write(folder, pre_tokenizer):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.save(str(folder / "tokenizer.json"))
    return folder


def pieces(tokenizer: Tokenizer, text: str) -> list[str]:
    return [piece for piece, _ in tokenizer.pre_tokenizer.pre_tokenize_str(text)]


def test_the_release_pattern_is_replaced(tmp_path):
    old = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(OLD_PATTERN), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    stored = Tokenizer.from_file(str(write(tmp_path, old) / "tokenizer.json"))
    loaded = load_tokenizer(tmp_path)
    # Marks stay with their letters: one piece per word, not split at each vowel sign.
    assert len(pieces(loaded, HINDI)) == 2
    assert len(pieces(stored, HINDI)) > 2


def test_other_pre_tokenizers_are_kept(tmp_path):
    loaded = load_tokenizer(write(tmp_path, pre_tokenizers.Whitespace()))
    assert pieces(loaded, "a <b> c") == ["a", "<", "b", ">", "c"]
