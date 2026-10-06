"""Write the extra parity requests for the Laya and Julia families.

usage: python scripts/make_marker_parity_set.py [--out tests/parity/marker/requests.json]

The marker families are checked on the text requests of the main parity set
(tests/parity/requests.json) plus these: multilingual states, the cases
their encoders cut or refuse (marker tokens in text, long options, many
options, long instructions, conversations), and states sized around Laya's
512- and 1,024-token limits. All texts are written for this project.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ROUTING = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this message?",
        "criteria": {
            "billing": "Payments, invoices, refunds",
            "technical": "Bugs, crashes, outages",
            "account": "Login, passwords, profile",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this message?",
        "criteria": ["Not urgent", "Soon", "Right away"],
    },
    "angry": {"type": "noul", "instructions": "The writer is angry."},
}

MULTILINGUAL = {
    "de": "Mir wurde die Rechnung für März zweimal abgebucht. Bitte erstatten Sie den "
    "doppelten Betrag heute noch, sonst kündige ich.",
    "fr": "L'application plante chaque fois que j'ouvre les paramètres. C'est très "
    "agaçant, j'en ai besoin pour mon travail.",
    "es": "No puedo iniciar sesión desde ayer; el código de verificación nunca llega a "
    "mi teléfono.",
    "pt": "Fui cobrado duas vezes pela mesma assinatura. Quero o reembolso imediatamente!",
    "it": "Il sito è lentissimo da stamattina e alcune pagine danno errore 500.",
    "nl": "Ik wil mijn wachtwoord wijzigen, maar de link in de e-mail werkt niet.",
    "pl": "Aplikacja zawiesza się przy każdym logowaniu. Proszę o szybką pomoc.",
    "tr": "Faturamda iki kez ücret alınmış, lütfen fazla ödenen tutarı iade edin.",
    "ru": "Приложение вылетает при открытии настроек. Очень неудобно, исправьте, пожалуйста.",
    "uk": "Не можу увійти до облікового запису: пароль не приймається, хоча він правильний.",
    "ar": "تم خصم المبلغ مرتين من بطاقتي هذا الشهر. أرجو إعادة المبلغ الزائد فوراً.",
    "he": "האפליקציה קורסת בכל פעם שאני פותח את ההגדרות. זה דחוף.",
    "hi": "मुझसे मार्च में दो बार शुल्क लिया गया, कृपया डुप्लिकेट राशि वापस करें।",
    "bn": "আমি গতকাল থেকে আমার অ্যাকাউন্টে লগ ইন করতে পারছি না।",
    "zh": "每次打开设置应用都会崩溃，请尽快修复，我工作需要用它。",
    "ja": "パスワードを再設定しようとしましたが、メールが届きません。",
    "ko": "이번 달 요금이 두 번 청구되었습니다. 환불해 주세요.",
    "th": "แอปค้างทุกครั้งที่ฉันพยายามเข้าสู่ระบบ ช่วยด่วนด้วย",
    "vi": "Tôi bị trừ tiền hai lần cho cùng một đơn hàng, vui lòng hoàn tiền.",
    "id": "Situs web tidak bisa dibuka sejak pagi dan menampilkan pesan kesalahan.",
    "sw": "Siwezi kuingia kwenye akaunti yangu tangu jana, nisaidieni tafadhali.",
    "el": "Η εφαρμογή κολλάει κάθε φορά που ανοίγω τις ρυθμίσεις.",
}

PARAGRAPH = (
    "The customer writes that the invoice for the last quarter was charged twice, that the "
    "support chat closed before anyone answered, and that the dashboard has shown an error "
    "since the update on Tuesday. They ask for a refund of the duplicate charge and a call "
    "back from someone who can explain what happened with the update."
)  # about 60 tokens


def sized(paragraphs: int) -> str:
    return "\n\n".join(f"{i + 1}. {PARAGRAPH}" for i in range(paragraphs))


def cases() -> list[dict]:
    out = [
        {"id": f"lang_{code}", "state": text, "questions": ROUTING}
        for code, text in MULTILINGUAL.items()
    ]
    out.append(
        {
            "id": "lang_mixed_object",
            "state": {"de": MULTILINGUAL["de"], "ja": MULTILINGUAL["ja"], "ticket": 4711},
            "questions": ROUTING,
        }
    )
    long_option = " ".join(["a long and detailed description of what this team does"] * 8)
    out += [
        {
            "id": "marker_tokens_in_text",
            "state": "Please route [MASK] this <mask> message.",
            "questions": {
                "team": {**ROUTING["team"], "instructions": "Which team, [MASK] or <mask>?"},
            },
        },
        {
            "id": "marker_token_in_option",
            "state": "Refund please.",
            "questions": {
                "team": {
                    **ROUTING["team"],
                    "criteria": {"billing": "refunds [MASK] <mask>", "other": "anything else"},
                }
            },
        },
        {
            "id": "option_over_48_tokens",
            "state": "My invoice is wrong.",
            "questions": {
                "team": {**ROUTING["team"], "criteria": {"billing": long_option, "other": "rest"}}
            },
        },
        {
            "id": "fifteen_long_options",
            "state": "My invoice is wrong.",
            "questions": {
                "team": {
                    **ROUTING["team"],
                    "criteria": {f"team_{i}": f"{long_option[:150]} ({i})" for i in range(15)},
                }
            },
        },
        {
            "id": "twenty_one_options",
            "state": "My invoice is wrong.",
            "questions": {
                "team": {
                    **ROUTING["team"],
                    "criteria": {f"team_{i}": f"Team number {i}" for i in range(21)},
                }
            },
        },
        {
            "id": "long_instructions",
            "state": "The app crashed twice today.",
            "questions": {
                "angry": {
                    "type": "noul",
                    "instructions": "Read the message carefully and decide. " * 40
                    + "Is the writer angry?",
                }
            },
        },
        {
            "id": "conversation_1k",
            "state": [f"{'customer' if i % 2 == 0 else 'agent'}: {PARAGRAPH}" for i in range(16)]
            + ["customer: Forget it, I am cancelling my subscription today."],
            "questions": {
                "churn": {"type": "noul", "instructions": "The customer says they will cancel."},
                "team": ROUTING["team"],
            },
        },
        {"id": "state_450", "state": sized(7), "questions": ROUTING},
        {"id": "state_600", "state": sized(10), "questions": ROUTING},
        {"id": "state_900", "state": sized(15), "questions": ROUTING},
        {"id": "state_1100", "state": sized(18), "questions": ROUTING},
    ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=ROOT / "tests" / "parity" / "marker" / "requests.json"
    )
    args = parser.parse_args()
    requests = cases()
    main_ids = {c["id"] for c in json.loads((ROOT / "tests/parity/requests.json").read_text())}
    ids = [case["id"] for case in requests]
    assert len(ids) == len(set(ids)) and not set(ids) & main_ids, "duplicate case ids"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(requests, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(requests)} requests to {args.out}")


if __name__ == "__main__":
    main()
