# Skill: pdfs

## Version
1.0.0

## Description
Inspect, extract, summarize, create, and validate PDF documents while preserving page-specific evidence and visual layout awareness.

## Triggers
- inspect this PDF
- summarize a PDF
- extract tables from a PDF
- create a PDF report
- validate PDF layout

## Negative Triggers
- create an Excel workbook
- make a presentation deck
- draft an email reply

## Required Tools
- pdf

## Allowed Tools
- pdf
- filesystem
- image-renderer

## Forbidden Tools
- email-send

## Preflight Steps
- Confirm the target PDF path exists before analysis.
- Determine whether visual layout, text extraction, or table extraction is required.

## Execution Steps
- Load the PDF with a structured PDF reader or renderer.
- Use page-aware extraction for text, tables, and visual findings.
- Render pages when layout or visual content matters.

## Hard Constraints
- Inspect the PDF before making claims about visual content.
- Cite page-specific findings when possible.
- Do not rely on OCR unless necessary.
- Validate generated PDF files before returning.

## Soft Guidelines
- Prefer concise summaries with page references.
- Separate extraction uncertainty from confirmed observations.

## Validation Steps
- Confirm the PDF file exists before reading or returning it.
- Render or reopen generated PDFs when layout matters.
- Check page count and key content after generation.

## Failure Modes
- Password-protected PDF.
- Scanned pages require OCR.
- Page coordinates are ambiguous.

## Examples
- user_request: Summarize this PDF and call out important tables.
  expected_behavior: Inspect the file, cite page-specific findings, and state when table extraction is uncertain.
- user_request: Create a two-page PDF report from this summary.
  expected_behavior: Generate the PDF, validate it exists, reopen or render it, and return a file link.

## Eval Cases
- name: pdf_visual_claims
  user_request: What visual issues are present in this PDF?
  expected_skills: pdfs
  required_constraints:
    - Inspect the PDF before making claims about visual content.
- name: pdf_page_citations
  user_request: Summarize this PDF with page citations.
  expected_skills: pdfs
  required_constraints:
    - Cite page-specific findings when possible.

## Metadata
{
  "domain": "documents",
  "artifact_types": ["pdf"]
}
