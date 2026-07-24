import base64
import json
import re

import pdfplumber
from anthropic import Anthropic
from fastapi import HTTPException, UploadFile

from schemas import BillAnalysis

client = Anthropic()  # reads ANTHROPIC_API_KEY from env

MODEL = "claude-sonnet-4-6"

IMAGE_MEDIA_TYPES = {
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/png": "image/png",
    "image/webp": "image/webp",
}

SYSTEM_PROMPT = """You are the audit engine behind CareClarify, a patient-advocacy tool that \
reviews Indian hospital bills and prescriptions for overcharges, duplicate billing, and \
regulatory-cap violations.

You will be given the text (or an image) of a real hospital bill or invoice. Your job:

1. Extract every distinct billed line item you can identify, with its name and billed amount \
(in INR).
2. For each item, estimate a "fair" amount using your general knowledge of typical Indian \
hospital tariffs and known regulatory caps, for example:
   - NPPA price ceilings on coronary stents (~Rs 38,250 for bare metal / drug-eluting caps vary \
by category) and orthopedic knee implants.
   - Typical room rent bands (general ward vs private vs deluxe/ICU).
   - Standard consultation/visit fee ranges for specialists.
   - Common duplicate-billing patterns (e.g. a flat "consumables kit" charge alongside itemized \
consumable lines; billing days after actual discharge).
   - Brand-name pharmacy items billed at full MRP where a generic equivalent is typically far \
cheaper.
   - Items that appear billed but were clearly not administered based on the record.
3. Classify each item's category as exactly one of: "Room Rent", "OT Fees", "Doctor Fees", \
"Consumables", "Pharmacy", "Special Services", "Diagnostics", "Implants", "Other".
4. Classify each item's status as exactly one of: "Reasonable", "Overcharged", \
"Duplicate Billed", "Capping Violation", "Not Administered", "Verify Charge".
   - Use "Reasonable" when the billed amount is at or close to your fair estimate.
   - Use "Verify Charge" only when you genuinely cannot judge fairness from the given \
information (do not overuse this).
5. Give each item a short, specific one-to-two sentence "reason" explaining the fair-price \
rationale, citing the relevant regulation or norm where applicable.
6. Identify the 2-4 most important findings across the whole bill as "warnings" a patient \
should act on. Each warning needs: a short id (e.g. "warn-1"), a type of "danger" (clear \
overcharge/violation) or "warning" (needs verification), a title, 1-3 sentence text explaining \
the issue, and a short imperative "action" label (e.g. "Flag NPPA Compliance Violation").
7. Fill in patient/hospital identifying fields ONLY if they literally appear in the source \
document. If a field is not present in the document, use a clearly generic placeholder like \
"Patient" or "Hospital / Facility Name" rather than inventing specifics — never fabricate a \
real-sounding name, date, or ID that isn't in the source.
8. Give the whole bill a short "title" summarizing the case (e.g. "General Ward Hospitalization \
Bill").

Be conservative and honest: if the document doesn't look like a hospital bill at all, or has too \
little detail to analyze, still return the best-effort JSON structure but keep items minimal and \
say so plainly in a single warning with type "warning".

Respond with ONLY a single JSON object — no markdown code fences, no preamble, no commentary — \
matching exactly this shape:

{
  "title": string,
  "patient": string,
  "date": string,
  "billNo": string,
  "hospitalName": string,
  "items": [
    {"name": string, "original": number, "fair": number, "category": string, "status": string, "reason": string}
  ],
  "warnings": [
    {"id": string, "type": "danger" | "warning", "title": string, "text": string, "action": string}
  ]
}

Do not include totals or chart data — those are computed by the caller from your items list."""


def _extract_pdf_text(data: bytes) -> str:
    text_parts = []
    with pdfplumber.open(__import__("io").BytesIO(data)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


async def analyze_bill_file(file: UploadFile, pasted_text: str | None) -> BillAnalysis:
    data = await file.read() if file is not None else b""
    content_type = (file.content_type or "").lower() if file is not None else ""

    user_content: list[dict] = []

    if pasted_text and pasted_text.strip():
        user_content.append({
            "type": "text",
            "text": f"Here is the bill text pasted by the patient:\n\n{pasted_text.strip()}",
        })
    elif content_type in IMAGE_MEDIA_TYPES:
        user_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": IMAGE_MEDIA_TYPES[content_type],
                "data": base64.b64encode(data).decode("utf-8"),
            },
        })
        user_content.append({
            "type": "text",
            "text": "Here is a photo/scan of the hospital bill. Extract and audit every line item you can read.",
        })
    elif content_type == "application/pdf":
        pdf_text = _extract_pdf_text(data)
        if not pdf_text:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Couldn't extract text from this PDF — it may be a scanned image with no "
                    "text layer. Please upload it as a JPEG/PNG photo instead, or paste the "
                    "bill text directly."
                ),
            )
        user_content.append({
            "type": "text",
            "text": f"Here is the extracted text of a hospital bill PDF:\n\n{pdf_text}",
        })
    elif content_type in ("text/plain",) or (file is not None and file.filename and file.filename.endswith(".txt")):
        user_content.append({
            "type": "text",
            "text": f"Here is a hospital bill as plain text:\n\n{data.decode('utf-8', errors='ignore')}",
        })
    else:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{content_type}'. Please upload a JPEG, PNG, PDF, or .txt file.",
        )

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"AI analysis failed: {exc}") from exc

    raw_text = "".join(block.text for block in response.content if block.type == "text")
    cleaned = _strip_json_fences(raw_text)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail="The AI returned a response that couldn't be parsed. Please try again.",
        ) from exc

    items = parsed.get("items", [])
    total_original = sum(float(i.get("original", 0)) for i in items)
    total_fair = sum(float(i.get("fair", 0)) for i in items)
    overcharge = round(total_original - total_fair, 2)

    chart_data = {"room": 0.0, "ot": 0.0, "doctor": 0.0, "consumables": 0.0, "other": 0.0}
    category_map = {
        "Room Rent": "room",
        "OT Fees": "ot",
        "Doctor Fees": "doctor",
        "Consumables": "consumables",
        "Implants": "consumables",
    }
    for item in items:
        bucket = category_map.get(item.get("category"), "other")
        chart_data[bucket] += float(item.get("fair", 0))

    result = {
        "title": parsed.get("title", "Hospital Bill Audit"),
        "patient": parsed.get("patient", "Patient"),
        "date": parsed.get("date", ""),
        "billNo": parsed.get("billNo", ""),
        "hospitalName": parsed.get("hospitalName", "Hospital / Facility Name"),
        "totalOriginal": round(total_original, 2),
        "totalFair": round(total_fair, 2),
        "overcharge": overcharge,
        "items": items,
        "warnings": parsed.get("warnings", []),
        "chartData": chart_data,
    }

    return BillAnalysis(**result)
