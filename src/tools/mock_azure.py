"""
Mock Azure Document Intelligence
---------------------------------
Simulates the Azure Document Intelligence prebuilt-check model
so the project runs without any Azure credentials or billing.

In production, replace this module with:

    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    client = DocumentIntelligenceClient(
        endpoint=os.getenv("AZURE_DOC_INTEL_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("AZURE_DOC_INTEL_KEY")),
    )
    poller = client.begin_analyze_document("prebuilt-check", ...)
    result = poller.result()
"""

import re
from dataclasses import dataclass


@dataclass
class MockField:
    content: str | None
    confidence: float


@dataclass
class MockDocumentResult:
    fields: dict[str, MockField]


def analyze_cheque(cheque_text: str) -> MockDocumentResult:
    """
    Extract basic fields from raw cheque text using simple regex heuristics.
    Mirrors the interface of the real Azure Document Intelligence response.

    Returns a MockDocumentResult with fields and mock confidence scores.
    """
    fields: dict[str, MockField] = {}

    # Amount — look for $ or AED followed by digits
    amount_match = re.search(r"(?:AED|\$)\s*([\d,]+(?:\.\d{2})?)", cheque_text)
    if amount_match:
        raw = amount_match.group(1).replace(",", "")
        fields["amount_numeric"] = MockField(content=raw, confidence=0.92)
    else:
        fields["amount_numeric"] = MockField(content=None, confidence=0.0)

    # Date — match common date patterns
    date_match = re.search(
        r"(?:Date[:\s]+)?(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}|\d{4}-\d{2}-\d{2})",
        cheque_text,
    )
    fields["cheque_date"] = MockField(
        content=date_match.group(1) if date_match else None,
        confidence=0.88 if date_match else 0.0,
    )

    # MICR — pattern: digits|digits|digits
    micr_match = re.search(r"MICR[:\s]+([\d\|]+)", cheque_text)
    fields["micr_line"] = MockField(
        content=micr_match.group(1) if micr_match else None,
        confidence=0.95 if micr_match else 0.0,
    )

    # Payee — look for "Pay to" or "Payee:" prefix
    payee_match = re.search(
        r"(?:Pay\s+to(?:\s+the\s+order\s+of)?|Payee\s*:)\s+([A-Z][A-Za-z\s]+?)(?:\s{2,}|\*|\$|AED|$)",
        cheque_text,
    )
    fields["payee_name"] = MockField(
        content=payee_match.group(1).strip() if payee_match else None,
        confidence=0.85 if payee_match else 0.0,
    )

    return MockDocumentResult(fields=fields)
