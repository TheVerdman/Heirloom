# Skill: slides

## Version
1.0.0

## Description
Create, inspect, edit, and validate presentation decks with concise slide structure and editable artifacts.

## Triggers
- make a slide deck
- create a presentation
- revise these slides
- build a keynote-style deck
- validate a PPTX file

## Negative Triggers
- analyze a spreadsheet
- extract PDF tables
- send an email

## Required Tools
- presentation

## Allowed Tools
- presentation
- filesystem
- image-renderer

## Forbidden Tools
- email-send

## Preflight Steps
- Confirm requested slide count, audience, and output format.
- Inspect any source deck before editing.

## Execution Steps
- Create or edit slides using presentation-native structures.
- Keep slide text concise and visually balanced.
- Preserve requested structure unless the user asks for a redesign.

## Hard Constraints
- Preserve requested slide count when specified.
- Use concise slide text.
- Validate that the presentation file exists before returning.
- Return an editable presentation artifact when creating a deck.

## Soft Guidelines
- Prefer strong titles, simple layouts, and consistent visual rhythm.
- Use speaker notes only when they add useful presenter context.

## Validation Steps
- Confirm the output PPTX exists on disk.
- Reopen or inspect the deck structure after writing.
- Verify slide count and major requested sections.

## Failure Modes
- Source deck is inaccessible.
- Requested slide count conflicts with supplied outline.
- Visual overflow on small slides.

## Examples
- user_request: Make a 6-slide investor update deck from this outline.
  expected_behavior: Preserve six slides, keep copy concise, validate the PPTX exists, and return a file link.
- user_request: Clean up this slide deck and keep the same number of slides.
  expected_behavior: Inspect the source deck, edit presentation-native elements, preserve slide count, and validate the result.

## Eval Cases
- name: slide_count_preserved
  user_request: Make a 5-slide product strategy presentation.
  expected_skills: slides
  required_constraints:
    - Preserve requested slide count when specified.
- name: editable_deck
  user_request: Create an editable PPTX from this launch outline.
  expected_skills: slides
  required_constraints:
    - Return an editable presentation artifact when creating a deck.

## Metadata
{
  "domain": "presentations",
  "artifact_types": ["pptx"]
}
