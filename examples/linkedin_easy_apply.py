#!/usr/bin/env python3
"""
Orbit Example — LinkedIn Easy Apply Bot

Automatically applies to jobs on LinkedIn using Easy Apply.
Uses Orbit's composable verbs (do, read, check, navigate) as short-horizon
tasks, with Python driving the state machine between them.

Usage:
    python linkedin_easy_apply.py --query "Software Engineer Intern" --count 10 --resume ~/Desktop/RESUME.pdf
    python linkedin_easy_apply.py --query "ML Engineer" --count 5 --resume resume.pdf --applicant profile.txt

The --applicant flag points to a text file with your info (one "- Key: Value" per line).
If omitted, a built-in placeholder profile is used — edit DEFAULT_APPLICANT_INFO below.
"""

from dotenv import load_dotenv

load_dotenv()

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

from pydantic import BaseModel, Field

import orbit

# Defaults 

LLM = "gemini-3-flash-preview"

DEFAULT_APPLICANT_INFO = """\
- Phone: 4083906345
- City: San Jose, CA
- LinkedIn: linkedin.com/in/example
- Website/Portfolio: example.com
- GitHub: github.com/example
- GPA: 3.8
- Graduation year: 2025
- Degree: Bachelor's in Computer Science
- University: San Jose State University
- Years of experience: 1
- Salary expectation: Open / 0
- Start date: ASAP
- Work authorization: Yes, authorized to work in the US
- Require sponsorship: No
- Willing to relocate: Yes
- Willing to commute: Yes
- Gender/Race/Veteran/Disability: Prefer not to say
- For any yes/no question about qualifications: Yes
- For any question about how you heard: LinkedIn
"""


# Exceptions 

class SkipJob(Exception):
    """Permanent skip — job is ineligible."""


class RetryJob(Exception):
    """Transient failure — retry without incrementing skipped."""


# Pydantic schemas (flat — cheap to read) 

class JobPanelState(BaseModel):
    """State after clicking a job in the left list."""
    job_title: str = Field(description="Title of the selected job")
    company: str = Field(description="Company name")
    has_easy_apply: bool = Field(
        description="Is an 'Easy Apply' button visible in the right panel?"
    )
    has_applied_badge: bool = Field(
        description="Does this job show an 'Applied' badge?"
    )


class ModalState(BaseModel):
    """State immediately after clicking Easy Apply."""
    modal_open: bool = Field(description="Is an Easy Apply modal/form currently open?")
    has_safety_popup: bool = Field(
        description="Is a safety reminder popup blocking the modal?"
    )


class WizardPageState(BaseModel):
    """State of the current Easy Apply wizard page.

    IMPORTANT: has_confirmation means a POST-SUBMISSION success message like
    'Your application was sent'. A 'Review your application' page is NOT confirmation.
    """
    has_submit_button: bool = Field(
        description="Is a 'Submit application' button visible? This is the FINAL submit, not 'Next'."
    )
    has_next_button: bool = Field(
        description="Is a 'Next', 'Continue', or 'Continue to next step' button visible?"
    )
    has_review_button: bool = Field(
        description="Is a 'Review' button visible?"
    )
    has_resume_upload: bool = Field(
        description="Is a resume upload control visible?"
    )
    resume_attached: bool = Field(
        description="Is a resume file already attached (filename shown)?"
    )
    has_confirmation: bool = Field(
        description="Is a SUCCESS message visible AFTER submission, like 'Your application was sent'? "
        "A review page is NOT confirmation."
    )
    has_unfilled_fields: bool = Field(
        description="Are there any empty/unfilled required form fields on this page?"
    )
    visible_buttons: list[str] = Field(description="All visible button labels")
    page_description: str = Field(
        description="Brief description of this wizard page, e.g. 'Contact info' or 'Additional Questions'"
    )


# Verb wrappers 
async def do(s, task: str, max_steps=15, extra_info=None, verbose=True):
    return await s.do(task, max_steps=max_steps, guidance=extra_info)


async def read(s, task: str, schema, max_steps=15):
    result = await s.read(task, schema=schema, max_steps=max_steps)
    return result.output


async def safe_read(s, task: str, schema, retries: int = 2, max_steps=15):
    for attempt in range(retries):
        try:
            result = await read(s, task, schema, max_steps=max_steps)
            if result is not None:
                return result
        except Exception as e:
            print(f"  [safe_read] Attempt {attempt + 1}/{retries} failed: {e}")
        if attempt < retries - 1:
            await asyncio.sleep(1)
    return None


async def check(s, condition: str) -> bool:
    return await s.check(condition, max_steps=13)


# Helpers 

async def close_modal(s):
    if await check(s, "A modal, dialog, or popup is currently open"):
        await do(s, "Close the modal by clicking the X button or Dismiss.",
                 max_steps=13, verbose=False)


async def scroll_job_list(s):
    await do(s,
        "Click any job title in the LEFT panel to focus it, "
        "then press PageDown 3 times.",
        max_steps=13, verbose=False,
    )


# Core: apply to one job 

async def apply_to_job(s, resume_path: str, applicant_info: str) -> bool:
    """Walk the Easy Apply wizard for the currently selected job.
    Returns True on successful submission.
    Raises SkipJob / RetryJob on failure."""

    # Step 1: Select an unapplied job 
    await do(s,
        "In the LEFT jobs list, click the TITLE text of a job that "
        "does NOT show 'Applied'. If all visible jobs show 'Applied', "
        "press PageDown to scroll and find one. "
        "Only click the title text — not any button.",
        max_steps=15,
    )

    panel: Optional[JobPanelState] = await safe_read(s,
        "the right job detail panel — title, company, and available buttons",
        schema=JobPanelState,
    )
    if panel is None:
        raise RetryJob("Could not read job panel state.")

    print(f"  Job: {panel.job_title} @ {panel.company} | Easy Apply: {panel.has_easy_apply}")

    if panel.has_applied_badge:
        raise SkipJob("Already applied.")
    if not panel.has_easy_apply:
        raise SkipJob("No Easy Apply button.")

    # Step 2: Open Easy Apply modal 
    await do(s,
        "Click the 'Easy Apply' button in the right panel. "
        "NOT 'Save', NOT the bookmark icon, NOT 'Apply' (external link).",
        extra_info=f"Job: {panel.job_title} at {panel.company}.",
    )

    modal: Optional[ModalState] = await safe_read(s,
        "the current screen — is an Easy Apply modal open?",
        schema=ModalState,
    )
    if modal is None:
        raise RetryJob("Could not read modal state.")

    print(f"  Modal open: {modal.modal_open}")

    if modal.has_safety_popup and not modal.modal_open:
        await do(s, "Dismiss the safety reminder popup by clicking Continue or Got it.",
                 max_steps=13, verbose=False)
        modal = await safe_read(s, "the current screen after dismissing popup", schema=ModalState)
        if modal is None:
            raise RetryJob("Could not read modal state after dismissing popup.")

    if not modal.modal_open:
        raise RetryJob("Modal did not open.")

    # Step 3: Walk the wizard 
    MAX_PAGES = 10
    last_desc = None

    for page_num in range(MAX_PAGES):
        page: Optional[WizardPageState] = await safe_read(s,
            "the current Easy Apply wizard page — what buttons are visible and are there unfilled fields?",
            schema=WizardPageState,
        )
        if page is None:
            await close_modal(s)
            raise RetryJob("Could not read wizard page state.")

        print(f"  Page {page_num + 1}: {page.page_description}")
        print(f"    Buttons: {page.visible_buttons} | Unfilled: {page.has_unfilled_fields}")

        # Submit (check FIRST) 
        if page.has_submit_button:
            await do(s,
                "Click the 'Submit application' button. "
                "Do NOT click 'Dismiss', 'Save', or 'X'.",
                max_steps=13,
                extra_info=f"Buttons: {page.visible_buttons}. Click ONLY 'Submit application'.",
            )
            final = await safe_read(s,
                "the screen after clicking submit — is there a success/confirmation message?",
                schema=WizardPageState,
            )
            await close_modal(s)
            if final is None:
                raise RetryJob("Could not verify submission confirmation.")
            return final.has_confirmation

        # Post-submission confirmation 
        if page.has_confirmation:
            await close_modal(s)
            return True

        # Loop-break guard 
        if page.page_description == last_desc:
            print("  !! Same page twice — stuck. Closing.")
            await close_modal(s)
            raise RetryJob("Wizard stuck on same page.")
        last_desc = page.page_description

        # Resume upload 
        if page.has_resume_upload and not page.resume_attached:
            await do(s,
                f"Upload the resume at {resume_path}. "
                "Use get_system_info to resolve the full path, then upload_file.",
                max_steps=15,
            )

        # Fill fields 
        if page.has_unfilled_fields:
            await do(s,
                "Fill ALL empty/unfilled required fields on this Easy Apply form page.\n"
                "USE THESE TOOLS (1 call per field — very efficient):\n"
                "  - type_into(pid, field_query, text) — for text/number fields\n"
                "  - select_dropdown_option(pid, dropdown_query, option) — for dropdowns\n"
                "  - select_option_by_label(pid, label_text) — for radio buttons / Yes-No\n"
                "First call list_active_windows() to get the browser PID, then fill each field.\n"
                "Do NOT click Next, Submit, Review, or any navigation button. Only fill fields.",
                max_steps=30,
                extra_info=(
                    f"Page: {page.page_description}\n"
                    f"Buttons (DO NOT CLICK): {page.visible_buttons}\n\n"
                    f"Applicant info to use:\n{applicant_info}"
                ),
            )

        # Advance 
        if page.has_next_button:
            await do(s,
                "Click the 'Next', 'Continue', or 'Continue to next step' button. "
                "Do NOT click 'Save', 'Review later', 'Back', or 'Dismiss'.",
                max_steps=13,
                extra_info=f"Buttons: {page.visible_buttons}",
            )
        elif page.has_review_button:
            await do(s,
                "Click the 'Review' button. "
                "Do NOT click 'Back', 'Save', or 'Dismiss'.",
                max_steps=13,
                extra_info=f"Buttons: {page.visible_buttons}",
            )
        else:
            # Submit may be below the fold — scroll and re-check
            print("  No visible button — scrolling down to check for Submit...")
            await do(s,
                "Scroll down inside the modal/dialog to reveal any buttons below the fold. "
                "Use scroll_page('down', 3) or press PageDown.",
                max_steps=5, verbose=False,
            )
            retry_page = await safe_read(s,
                "the current Easy Apply wizard page after scrolling — any Submit button now?",
                schema=WizardPageState,
            )
            if retry_page and retry_page.has_submit_button:
                await do(s,
                    "Click the 'Submit application' button. "
                    "Do NOT click 'Dismiss', 'Save', or 'X'.",
                    max_steps=13,
                    extra_info=f"Buttons: {retry_page.visible_buttons}. Click ONLY 'Submit application'.",
                )
                final = await safe_read(s,
                    "the screen after clicking submit — is there a success/confirmation message?",
                    schema=WizardPageState,
                )
                await close_modal(s)
                if final is None:
                    raise RetryJob("Could not verify submission confirmation.")
                return final.has_confirmation
            else:
                print("  Still no button after scroll — closing.")
                await close_modal(s)
                return False

    print("  Exhausted max pages without submitting.")
    await close_modal(s)
    return False


# Main loop 

async def run(query: str, count: int, resume_path: str, applicant_info: str):
    search_url = (
        f"https://www.linkedin.com/jobs/search/"
        f"?f_AL=true&keywords={quote_plus(query)}"
    )

    applied = 0
    skipped = 0
    consecutive_skips = 0

    # Single session for the entire run — no re-navigation between jobs.
    async with orbit.session(llm=LLM) as s:
        # Navigate once
        print(f"\n  Navigating to: {query}")
        await s.navigate(search_url, max_steps=15)

        while applied < count:
            attempt = applied + skipped + 1
            print(f"\n{'='*60}")
            print(f"  Attempt {attempt}  |  Applied: {applied}/{count}  |  Skipped: {skipped}")
            print(f"{'='*60}")

            try:
                success = await apply_to_job(s, resume_path, applicant_info)
                if success:
                    applied += 1
                    consecutive_skips = 0
                    print(f"\n  Applied! ({applied}/{count})")
                else:
                    skipped += 1
                    consecutive_skips += 1
                    print(f"\n  Skipped. ({skipped} total)")

            except SkipJob as e:
                skipped += 1
                consecutive_skips += 1
                print(f"\n  Skipped (ineligible): {e}")

            except RetryJob as e:
                consecutive_skips += 1
                print(f"\n  Retrying (transient): {e}")

            except Exception as e:
                skipped += 1
                consecutive_skips += 1
                print(f"\n  Hard error — skipping: {e}")

            # If we've skipped many in a row, scroll to find fresh jobs
            if consecutive_skips >= 3:
                print("  Scrolling to find more jobs...")
                await scroll_job_list(s)
                consecutive_skips = 0

            # Safety valve — don't loop forever
            if skipped > count * 3:
                print(f"\n  Too many skips ({skipped}). Stopping.")
                break

    print(f"\n{'='*60}")
    print(f"  Done. Applied to {applied} jobs. ({skipped} skipped)")
    print(f"{'='*60}")


# CLI 

def main():
    parser = argparse.ArgumentParser(
        description="Orbit — LinkedIn Easy Apply bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --query "Software Engineer Intern" --count 10 --resume ~/Desktop/RESUME.pdf
  %(prog)s --query "ML Engineer" --count 5 --resume resume.pdf --applicant my_profile.txt
  %(prog)s -q "Data Scientist" -n 3 -r resume.pdf
""",
    )
    parser.add_argument(
        "-q", "--query", required=True,
        help="Job search query (e.g. 'Software Engineer Intern')",
    )
    parser.add_argument(
        "-n", "--count", type=int, default=10,
        help="Number of applications to submit (default: 10)",
    )
    parser.add_argument(
        "-r", "--resume", required=True,
        help="Path to resume file (PDF)",
    )
    parser.add_argument(
        "-a", "--applicant", type=str, default=None,
        help="Path to applicant info text file (one '- Key: Value' per line). "
             "If omitted, uses built-in defaults — edit DEFAULT_APPLICANT_INFO in the script.",
    )
    parser.add_argument(
        "--llm", type=str, default=None,
        help="LLM model to use (default: gemini-3-flash-preview)",
    )

    args = parser.parse_args()

    # Resolve resume path
    resume = Path(args.resume).expanduser()
    if not resume.exists():
        print(f"Error: Resume not found at {resume}", file=sys.stderr)
        sys.exit(1)

    # Load applicant info
    if args.applicant:
        applicant_path = Path(args.applicant).expanduser()
        if not applicant_path.exists():
            print(f"Error: Applicant file not found at {applicant_path}", file=sys.stderr)
            sys.exit(1)
        applicant_info = applicant_path.read_text()
    else:
        applicant_info = DEFAULT_APPLICANT_INFO

    # Override LLM if specified
    if args.llm:
        global LLM
        LLM = args.llm

    print(f"\n  Orbit — LinkedIn Easy Apply")
    print(f"  Query:   {args.query}")
    print(f"  Count:   {args.count}")
    print(f"  Resume:  {resume}")
    print(f"  LLM:     {LLM}")

    asyncio.run(run(
        query=args.query,
        count=args.count,
        resume_path=str(resume),
        applicant_info=applicant_info,
    ))


if __name__ == "__main__":
    main()
