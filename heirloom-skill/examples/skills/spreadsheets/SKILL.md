# Skill: spreadsheets

## Version
1.0.0

## Description
Create, inspect, repair, and validate spreadsheet artifacts such as CSV, TSV, and XLSX files using spreadsheet-native tooling.

## Triggers
- turn this CSV into a styled Excel dashboard
- create a spreadsheet report
- analyze this XLSX workbook
- clean spreadsheet data
- add formulas and charts to a workbook

## Negative Triggers
- edit a PDF layout
- make a slide deck
- send an email

## Required Tools
- spreadsheet

## Allowed Tools
- spreadsheet
- filesystem
- chart-renderer

## Forbidden Tools
- email-send

## Preflight Steps
- Inspect the input file shape and sheet names before transforming data.
- Confirm the requested output format and any chart or formula requirements.

## Execution Steps
- Load spreadsheet data with a structured spreadsheet reader.
- Preserve source columns unless the user requests a reshape.
- Apply formulas, tables, charts, and formatting through spreadsheet-native APIs.

## Hard Constraints
- Use spreadsheet-native tooling.
- Validate generated files before returning.
- Include a downloadable file link when producing an artifact.
- Do not claim a file was created unless it exists.

## Soft Guidelines
- Prefer compact workbook layouts that are easy to scan.
- Use clear sheet names and freeze header rows when useful.

## Validation Steps
- Verify the output workbook or CSV exists on disk.
- Reopen the produced spreadsheet artifact and inspect expected sheets.
- Confirm formulas, charts, and row counts match the requested transformation.

## Failure Modes
- Missing source file.
- Unsupported workbook format.
- Formula references break after a reshape.

## Examples
- user_request: Turn this CSV into a styled Excel dashboard.
  expected_behavior: Inspect the CSV, create a workbook with tables and charts, validate the file exists, and return a file link.
  notes: The final response should not claim success before validation.
- user_request: Clean this workbook and add a summary sheet.
  expected_behavior: Use spreadsheet-native operations, preserve raw data, add a summary sheet, and validate the saved workbook.

## Eval Cases
- name: spreadsheet_dashboard_route
  user_request: Turn this CSV into a styled Excel dashboard.
  expected_skills: spreadsheets
  required_constraints:
    - Use spreadsheet-native tooling.
    - Validate generated files before returning.
- name: spreadsheet_artifact_link
  user_request: Build an XLSX report with charts from these sales rows.
  expected_skills: spreadsheets
  required_constraints:
    - Include a downloadable file link when producing an artifact.

## Metadata
{
  "domain": "artifact_generation",
  "artifact_types": ["csv", "xlsx"]
}
