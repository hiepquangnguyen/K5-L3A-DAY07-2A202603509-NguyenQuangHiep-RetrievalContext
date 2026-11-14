"""Build the Day 7 Basic Benchmark (dev + independent held-out test + unanswerable cases) from exported Day 6 chunks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))

from baseline import hit, validate  # noqa: E402

DATA_DIR = ROOT / "starter" / "data" / "basic"


def build_benchmark() -> dict:
    chunks_path = DATA_DIR / "chunks.json"
    day6_q_path = DATA_DIR / "day6_questions_converted.json"
    if not chunks_path.exists() or not day6_q_path.exists():
        raise FileNotFoundError("Run src/export_day6.py first to export chunks.json and day6_questions_converted.json.")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    day6_questions = json.loads(day6_q_path.read_text(encoding="utf-8"))

    # Move all 20 Day 6 questions into dev (because Day 6 test questions were inspected in Day 6 report),
    # and fix q19 multi-chunk gold quotes to match the exact single-chunk substrings cleanly.
    answerable_questions: list[dict] = []
    for q in day6_questions:
        item = dict(q)
        item["split"] = "dev"
        if item["id"] == "q19":
            item["gold"] = [
                {
                    "source_id": "tt18-2021",
                    "page": 13,
                    "evidence": "Cha, mẹ, vợ hoặc chồng, con, anh, chị, em ruột của nghiên cứu sinh không tham gia Hội đồng.",
                },
                {
                    "source_id": "tt18-2021",
                    "page": 13,
                    "evidence": "Trong thời gian tối đa 03 tháng kể từ khi luận án của nghiên cứu sinh đạt quy trình phản biện độc lập",
                },
            ]
            item["multi_evidence"] = True
        else:
            item["multi_evidence"] = False
        answerable_questions.append(item)

    # Fresh independent held-out test questions on previously untouched pages (tt08 p.4,9,16; tt18 p.5,6,17).
    fresh_test = [
        {
            "id": "q21",
            "split": "test",
            "question": "Theo Quy chế đào tạo trình độ đại học, liên kết đào tạo chỉ được thực hiện đối với hình thức học nào và bị cấm đối với nhóm ngành nào?",
            "gold": [
                {
                    "source_id": "tt08-2021",
                    "page": 4,
                    "evidence": "Liên kết đào tạo chỉ thực hiện đối với hình thức vừa làm vừa học theo quy định tại khoản 22 Điều 1 Luật sửa đổi, bổ sung một số điều của Luật Giáo dục đại học và theo quy định tại khoản 2, khoản 3 Điều này. Không thực hiện việc liên kết đào tạo đối với các ngành thuộc lĩnh vực sức khoẻ có cấp chứng chỉ hành nghề.",
                }
            ],
            "expected_answer": "Liên kết đào tạo chỉ thực hiện đối với hình thức vừa làm vừa học và không được thực hiện đối với các ngành thuộc lĩnh vực sức khoẻ có cấp chứng chỉ hành nghề.",
            "filter": {"cap_dao_tao": "dai_hoc"},
            "multi_evidence": False,
        },
        {
            "id": "q22",
            "split": "test",
            "question": "Khi cơ sở đào tạo đại học cho phép sinh viên thi lại hoặc đánh giá lại một điểm thành phần để cải thiện điểm học phần từ không đạt thành đạt, điểm học phần sau khi thi lại bị giới hạn tối đa ở mức điểm chữ nào?",
            "gold": [
                {
                    "source_id": "tt08-2021",
                    "page": 9,
                    "evidence": "Việc cho phép thi lại , đánh giá lại một điểm thành phần để cải thiện điểm học phần từ không đạt thành đạt; trong trường hợp này, điểm học phần sau khi thi lại, đánh giá lại chỉ được giới hạn ở mức điểm D hoặc điểm C;",
                }
            ],
            "expected_answer": "Điểm học phần sau khi thi lại, đánh giá lại một điểm thành phần để cải thiện từ không đạt thành đạt chỉ được giới hạn ở mức điểm D hoặc điểm C.",
            "multi_evidence": False,
        },
        {
            "id": "q23",
            "split": "test",
            "question": "Trong trao đổi sinh viên đại học, số lượng tín chỉ tối đa mà sinh viên được phép tích lũy tại cơ sở đào tạo khác chiếm không quá bao nhiêu phần trăm tổng khối lượng chương trình đào tạo?",
            "gold": [
                {
                    "source_id": "tt08-2021",
                    "page": 16,
                    "evidence": "số lượng tín chỉ mà sinh viên tích lũy tại cơ sở đào tạo khác (nếu có) không vượt quá 25% tổng khối lượng của chương trình đào tạo.",
                }
            ],
            "expected_answer": "Số lượng tín chỉ mà sinh viên tích lũy tại cơ sở đào tạo khác không vượt quá 25% tổng khối lượng của chương trình đào tạo.",
            "multi_evidence": False,
        },
        {
            "id": "q24",
            "split": "test",
            "question": "Tại cùng một thời điểm, giảng viên có chức danh Giáo sư, Phó Giáo sư và Tiến sĩ được hướng dẫn độc lập tối đa bao nhiêu nghiên cứu sinh, và việc đồng hướng dẫn được quy đổi như thế nào?",
            "gold": [
                {
                    "source_id": "tt18-2021",
                    "page": 5,
                    "evidence": "Tại cùng một thời điểm, người có chức danh giáo sư được hướ ng dẫn độc lập tối đa 07 nghiên cứu sinh; ngườ i có ch ức danh phó giáo sư, hoặc có b ằng tiến sĩ khoa học được hướ ng dẫn độc lập tối đa 05 nghiên cứu sinh; người có b ằng tiến sĩ được hướ ng dẫn độc lập tối đa 03 nghiên cứu sinh. T rường hợp đồng hướ ng dẫn 01 nghiên c ứu sinh đượ c tính quy đổi tương đương hướng dẫn độc lập 0,5 nghiên cứu sinh.",
                }
            ],
            "expected_answer": "Giáo sư được hướng dẫn độc lập tối đa 07 nghiên cứu sinh, Phó Giáo sư hoặc Tiến sĩ khoa học tối đa 05 nghiên cứu sinh, Tiến sĩ tối đa 03 nghiên cứu sinh; đồng hướng dẫn 01 nghiên cứu sinh được quy đổi tương đương hướng dẫn độc lập 0,5 nghiên cứu sinh.",
            "filter": {"cap_dao_tao": "tien_si"},
            "multi_evidence": False,
        },
        {
            "id": "q25",
            "split": "test",
            "question": "Người dự tuyển trình độ tiến sĩ là công dân nước ngoài nếu đăng ký học bằng tiếng Việt phải có chứng chỉ tiếng Việt tối thiểu từ bậc mấy?",
            "gold": [
                {
                    "source_id": "tt18-2021",
                    "page": 6,
                    "evidence": "phải có chứng chỉ tiếng Việt tối thiểu từ bậc 4 trở lên theo Khung năng lực tiếng Việt dùng cho người nước ngoài",
                }
            ],
            "expected_answer": "Người dự tuyển là công dân nước ngoài đăng ký học trình độ tiến sĩ bằng tiếng Việt phải có chứng chỉ tiếng Việt tối thiểu từ bậc 4 trở lên theo Khung năng lực tiếng Việt dùng cho người nước ngoài (trừ người bản ngữ).",
            "multi_evidence": False,
        },
        {
            "id": "q26",
            "split": "test",
            "question": "Khi chất lượng luận án tiến sĩ không đạt yêu cầu thẩm định, Hội đồng thẩm định luận án do Bộ Giáo dục và Đào tạo thành lập gồm bao nhiêu thành viên và có tối thiểu bao nhiêu thành viên không là phản biện độc lập hoặc thành viên Hội đồng đánh giá luận án?",
            "gold": [
                {
                    "source_id": "tt18-2021",
                    "page": 17,
                    "evidence": "Hội đồng thẩm định có 07 thành viên , trong đó có tối thiểu 03 thành viên không là phản biện độc lập hoặc thành viên Hội đồng đánh giá luận án",
                }
            ],
            "expected_answer": "Hội đồng thẩm định luận án có 07 thành viên, trong đó có tối thiểu 03 thành viên không là phản biện độc lập hoặc thành viên Hội đồng đánh giá luận án.",
            "multi_evidence": False,
        },
    ]
    answerable_questions.extend(fresh_test)

    # Validate against starter/baseline.py contract on both splits independently.
    dev_q = [q for q in answerable_questions if q["split"] == "dev"]
    test_q = [q for q in answerable_questions if q["split"] == "test"]
    validate(chunks, dev_q)
    validate(chunks, test_q)

    # Unanswerable near-domain cases (verified absent across all 210 chunks).
    unanswerable_cases = [
        {
            "id": "u01",
            "split": "dev",
            "question": "Theo Quy chế đào tạo trình độ đại học và tiến sĩ trong bộ tài liệu, mức tiền học bổng khuyến khích học tập và điều kiện điểm tối thiểu để nhận học bổng là bao nhiêu?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Đã kiểm tra toàn bộ 210 chunks (TT08/2021 và TT18/2021): từ khóa 'học bổng' xuất hiện 0 lần; các quy chế đào tạo này không quy định chính sách hay mức tiền học bổng.",
        },
        {
            "id": "u02",
            "split": "dev",
            "question": "Quy chế đào tạo trình độ đại học quy định cách chấm điểm rèn luyện sinh viên theo từng học kỳ trên thang điểm bao nhiêu và xếp loại rèn luyện ra sao?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Đã kiểm tra toàn bộ 210 chunks: cụm 'điểm rèn luyện' xuất hiện 0 lần; TT08/2021 chỉ quy định đánh giá học phần và học lực (Điều 9–10).",
        },
        {
            "id": "u03",
            "split": "test",
            "question": "Theo Quy chế tuyển sinh và đào tạo trình độ tiến sĩ, mức thù lao hoặc lệ phí thẩm định, phản biện độc lập luận án tiến sĩ là bao nhiêu đồng?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Đã kiểm tra toàn bộ 210 chunks: các từ khóa 'thù lao', 'lệ phí' xuất hiện 0 lần; Điều 15 TT18/2021 chỉ quy định quy trình phản biện độc lập, không quy định mức chi phí tài chính.",
        },
        {
            "id": "u04",
            "split": "test",
            "question": "Quy chế đào tạo trình độ đại học có quy định điều kiện ưu tiên bố trí chỗ ở ký túc xá nội trú cho sinh viên năm nhất không?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Đã kiểm tra toàn bộ 210 chunks: các từ khóa 'ký túc xá', 'nội trú' xuất hiện 0 lần; TT08/2021 không điều chỉnh công tác nội trú sinh viên.",
        },
    ]

    # Verify absence keywords in chunks directly.
    for kw in ("học bổng", "điểm rèn luyện", "thù lao", "lệ phí", "ký túc xá", "nội trú"):
        if any(kw in c["text"].lower() for c in chunks):
            raise AssertionError(f"Unexpected keyword match in corpus for unanswerable guard: {kw}")

    cases = [
        {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer",
            **({"filter": q["filter"]} if "filter" in q else {}),
        }
        for q in answerable_questions
    ] + unanswerable_cases

    questions_out = DATA_DIR / "questions.json"
    cases_out = DATA_DIR / "cases.json"
    questions_out.write_text(json.dumps(answerable_questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cases_out.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Verify every gold entry resolves cleanly.
    for q in answerable_questions:
        for g in q["gold"]:
            assert any(hit(c, [g]) for c in chunks), f"Unresolved gold for {q['id']}"

    summary = {
        "questions_path": str(questions_out),
        "cases_path": str(cases_out),
        "dev_answerable": len(dev_q),
        "dev_unanswerable": sum(1 for c in unanswerable_cases if c["split"] == "dev"),
        "test_answerable": len(test_q),
        "test_unanswerable": sum(1 for c in unanswerable_cases if c["split"] == "test"),
        "total_cases": len(cases),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    build_benchmark()
