# Skill: email

## Version
1.0.0

## Description
Draft, revise, and prepare emails while preserving recipient intent, tone, and attachment truthfulness.

## Triggers
- draft an email
- reply to this email
- make this email warmer
- write a follow-up email
- summarize this thread into a response

## Negative Triggers
- create a PDF report
- build a slide deck
- edit source code

## Required Tools
- text-editor

## Allowed Tools
- text-editor
- contact-context

## Forbidden Tools
- email-send

## Preflight Steps
- Identify recipient, sender intent, and desired tone.
- Confirm whether the user asked for a draft or explicitly requested sending.

## Execution Steps
- Draft email text that matches the requested tone.
- Preserve recipient intent and avoid adding unsupported commitments.
- Treat sending as a separate explicit action.

## Hard Constraints
- Preserve recipient intent.
- Draft instead of sending unless the user explicitly requests sending.
- Keep tone aligned with the user's request.
- Do not invent attachments or recipients.

## Soft Guidelines
- Prefer clear subject lines and concise opening context.
- Offer variants only when tone or audience is ambiguous.

## Validation Steps
- Check that recipient, tone, and requested action are reflected.
- Verify no invented attachments, recipients, dates, or commitments were added.
- Confirm the output is a draft unless sending was explicitly requested.

## Failure Modes
- Missing recipient context.
- Ambiguous requested tone.
- User asks to send without confirming final content.

## Examples
- user_request: Draft a warm follow-up email to Maya about the proposal.
  expected_behavior: Produce a draft only, keep the tone warm, preserve the proposal context, and avoid inventing attachments.
- user_request: Rewrite this email to sound firmer but still kind.
  expected_behavior: Preserve the sender's intent, adjust tone, and avoid adding unsupported facts.

## Eval Cases
- name: email_draft_not_send
  user_request: Draft a follow-up email to Alex about the meeting.
  expected_skills: email
  required_constraints:
    - Draft instead of sending unless the user explicitly requests sending.
- name: email_no_invented_attachments
  user_request: Reply to this customer and mention the attached invoice only if it exists.
  expected_skills: email
  required_constraints:
    - Do not invent attachments or recipients.

## Metadata
{
  "domain": "communication",
  "artifact_types": ["email_draft"]
}
