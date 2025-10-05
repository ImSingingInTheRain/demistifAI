import base64
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="demistifAI — Spam Detector", page_icon="📧", layout="wide")

CLASSES = ["spam", "safe"]
AUTONOMY_LEVELS = [
    "Low autonomy (predict + confidence)",
    "Moderate autonomy (recommendation)",
    "Full autonomy (auto-route)",
]

STARTER_LABELED: List[Dict] = [
    # ----------------------- SPAM (50) -----------------------
    {"title": "URGENT: Verify your payroll information today", "body": "Your salary deposit is on hold. Confirm your bank details via this external link to avoid delay.", "label": "spam"},
    {"title": "WIN a FREE iPhone — final round eligibility", "body": "Congratulations! Complete the short survey to claim your iPhone. Offer expires in 2 hours.", "label": "spam"},
    {"title": "Password will expire — action required", "body": "Reset your password here: http://accounts-security.example-reset.com. Failure to act may lock your account.", "label": "spam"},
    {"title": "Delivery notice: package waiting for customs clearance", "body": "Pay a small fee to release your parcel. Use our quick checkout link to avoid return.", "label": "spam"},
    {"title": "Your account was suspended", "body": "Unusual login detected. Open the attached HTML and confirm your credentials to restore access.", "label": "spam"},
    {"title": "Invoice discrepancy for March", "body": "We overcharged your account. Provide your card number for an immediate refund.", "label": "spam"},
    {"title": "Corporate survey: guaranteed €100 voucher", "body": "Finish this 1-minute survey and receive a voucher instantly. No employee ID needed.", "label": "spam"},
    {"title": "Security alert from IT desk", "body": "We’ve updated our security policy. Download the attached ZIP to review the changes.", "label": "spam"},
    {"title": "Limited-time premium subscription at 90% off", "body": "Upgrade now to unlock executive insights. Click the promotional link and pay today.", "label": "spam"},
    {"title": "Payment overdue: settle immediately", "body": "Your service will be interrupted. Wire funds to the account in the attachment to avoid penalties.", "label": "spam"},
    {"title": "HR update: bonus eligibility check", "body": "Confirm your identity by entering your national ID on our verification page to receive your bonus.", "label": "spam"},
    {"title": "Conference invite: free registration + gift", "body": "Register via the link to receive a €50 gift card. Limited seats; confirm with your credit card.", "label": "spam"},
    {"title": "DocuSign: You received a secure document", "body": "Open this third-party portal to review and log in with your email password for access.", "label": "spam"},
    {"title": "Crypto opportunity: double your balance", "body": "Transfer funds to our wallet and we’ll return 2× within 24 hours. Trusted by leaders.", "label": "spam"},
    {"title": "Password reset required immediately", "body": "We noticed unusual activity. Reset at http://security-reset.example-login.net to avoid permanent lock.", "label": "spam"},
    {"title": "Delivery fee required for redelivery", "body": "Your parcel is pending. Pay a €2.99 fee to schedule a new delivery slot.", "label": "spam"},
    {"title": "Payroll correction: refund available", "body": "Send your IBAN and CVV to process your refund now.", "label": "spam"},
    {"title": "COVID relief grant for employees", "body": "Claim your €500 benefit by verifying your bank details today.", "label": "spam"},
    {"title": "Urgent compliance training overdue", "body": "Access the training via our partner portal. Use your email password to sign in.", "label": "spam"},
    {"title": "Security notice: leaked credentials", "body": "Your email was found in a breach. Download the spreadsheet and confirm your password inside.", "label": "spam"},
    {"title": "Special discount on executive coaching", "body": "90% off today only. Pay via the link to secure your slot.", "label": "spam"},
    {"title": "Invoice correction required", "body": "We can refund you now if you send your card number and CVV for verification.", "label": "spam"},
    {"title": "IT ticket auto-closure warning", "body": "To keep your ticket open, log into the external support site and confirm your identity.", "label": "spam"},
    {"title": "One-time password: verify to unlock account", "body": "Enter the OTP on our portal along with your email password to restore access.", "label": "spam"},
    {"title": "Conference pass sponsored — confirm card", "body": "You’ve been awarded a free pass. Confirm your credit card to activate sponsorship.", "label": "spam"},
    {"title": "VPN certificate expired", "body": "Download the new certificate from our public link and install today.", "label": "spam"},
    {"title": "Tax rebate waiting", "body": "We owe you €248. Submit your bank details on our secure (external) claim page.", "label": "spam"},
    {"title": "CEO request: urgent payment", "body": "Transfer €4,900 immediately to the vendor in the attached invoice. Do not call.", "label": "spam"},
    {"title": "Doc review access restricted", "body": "Log in with Office365 password here to unlock the secure document.", "label": "spam"},
    {"title": "Account verification — final warning", "body": "Your mailbox will be closed. Verify now at http://mailbox-verify.example.org.", "label": "spam"},
    {"title": "Prize draw: employee appreciation", "body": "All staff eligible. Enter with your bank card details to receive your prize.", "label": "spam"},
    {"title": "Password reset confirmation (external)", "body": "Confirm reset by clicking this link and entering your credentials to finalize.", "label": "spam"},
    {"title": "Secure file delivered", "body": "Open the HTML attachment, enable macros, and sign in to download the file.", "label": "spam"},
    {"title": "Upgrade your mailbox storage", "body": "Pay €1 via micro-transaction to extend your mailbox by 50GB.", "label": "spam"},
    {"title": "Two-factor disabled", "body": "We turned off MFA on your account. Click the link to re-enable (login required).", "label": "spam"},
    {"title": "SaaS subscription renewal failed", "body": "Provide your card details to avoid losing access to premium features.", "label": "spam"},
    {"title": "Payroll bonus — confirm identity", "body": "Submit your personal ID and mobile TAN on the page to receive your bonus.", "label": "spam"},
    {"title": "Password policy update", "body": "Download the attached PDF from an unknown sender to review mandatory changes.", "label": "spam"},
    {"title": "Company survey — guaranteed gift card", "body": "Complete the survey; a €50 card will be emailed to you instantly after verification.", "label": "spam"},
    {"title": "License key expired", "body": "Activate your software by installing the attached executable and signing in.", "label": "spam"},
    {"title": "Unpaid toll invoice", "body": "Settle your unpaid toll by providing card details at the link below.", "label": "spam"},
    {"title": "Security incident report needed", "body": "Fill in this external Google Form with your employee credentials.", "label": "spam"},
    {"title": "Bank verification needed", "body": "We detected a failed withdrawal. Confirm your bank access to continue.", "label": "spam"},
    {"title": "Password rotation overdue", "body": "Rotate your password here: http://corp-passwords-reset.me to keep access.", "label": "spam"},
    {"title": "Delivery confirmation required", "body": "Click to pay a small customs fee and confirm your address to receive your parcel.", "label": "spam"},
    {"title": "Executive webinar: instant access", "body": "Reserve now; pay via partner portal and log in with your email credentials.", "label": "spam"},
    {"title": "Mail storage full", "body": "To avoid message loss, sign into the storage recovery portal with your password.", "label": "spam"},
    {"title": "Prize payout — action needed", "body": "We are ready to wire your winnings. Send IBAN and CVV to complete transfer.", "label": "spam"},
    {"title": "Compliance escalation", "body": "You are out of compliance. Download the attached document and log in to resolve.", "label": "spam"},
    {"title": "Tax invoice attached — payment portal", "body": "Open the link to pay immediately; failure to do so results in penalties.", "label": "spam"},

    # ----------------------- SAFE (50) -----------------------
    {"title": "Password reset confirmation for corporate portal", "body": "You requested a password change via the internal IT portal. If this wasn’t you, contact IT on Teams.", "label": "safe"},
    {"title": "DHL tracking: package out for delivery", "body": "Your parcel is scheduled today. Track via the official DHL site with your tracking ID (no payment required).", "label": "safe"},
    {"title": "March invoice attached — Accounts Payable", "body": "Hi, please find the March invoice attached in PDF. PO referenced below; no payment info requested.", "label": "safe"},
    {"title": "Minutes from the compliance workshop", "body": "Attached are the minutes and action items agreed during today’s workshop. Feedback welcome by Friday.", "label": "safe"},
    {"title": "Security advisory — internal policy update", "body": "Please review the new password guidelines on the intranet. No external links or attachments.", "label": "safe"},
    {"title": "Reminder: Q4 budget review on Tuesday", "body": "Please upload your cost worksheets to the internal SharePoint before 16:00 CEST.", "label": "safe"},
    {"title": "Travel itinerary update", "body": "Your train platform changed. Updated PDF itinerary is attached; no action required.", "label": "safe"},
    {"title": "Team meeting moved to 14:00", "body": "Join via the regular Teams link. Agenda: quarterly KPIs, risk register, and roadmap.", "label": "safe"},
    {"title": "Draft for review — policy document", "body": "Could you review the attached draft and add comments in Word track changes by EOD Thursday?", "label": "safe"},
    {"title": "Onboarding checklist for new starters", "body": "HR checklist attached; forms to be submitted via Workday. Reach out if anything is unclear.", "label": "safe"},
    {"title": "Canteen menu and wellness events", "body": "This week’s healthy menu and yoga session schedule are included. No RSVP needed.", "label": "safe"},
    {"title": "Customer feedback summary — Q3", "body": "Please see the attached slide deck with survey trends and next steps for CX improvements.", "label": "safe"},
    {"title": "Security alert — new device sign-in (confirmed)", "body": "You signed in from a new laptop. If recognized, no action needed. Audit log available on the intranet.", "label": "safe"},
    {"title": "Coffee catch-up?", "body": "Are you free on Thursday afternoon for a quick chat about the training roadmap?", "label": "safe"},
    {"title": "Password rotation reminder", "body": "This is an automated reminder to rotate your password on the internal portal this month.", "label": "safe"},
    {"title": "Delivery rescheduled by courier", "body": "Your parcel will arrive tomorrow morning. No fees due; track on the official courier portal.", "label": "safe"},
    {"title": "Payroll update posted", "body": "Payslips are available on the HR portal. Do not email personal details; use the secure site.", "label": "safe"},
    {"title": "COVID-19 office guidance", "body": "Updated office access rules are published on the intranet. Masks optional in open areas.", "label": "safe"},
    {"title": "Compliance training enrolled", "body": "You have been enrolled in the mandatory training via the LMS. Deadline next Friday.", "label": "safe"},
    {"title": "Security bulletin — phishing simulation next week", "body": "IT will run a phishing simulation. Do not click unknown links; report suspicious emails via the button.", "label": "safe"},
    {"title": "Corporate survey — help improve the office", "body": "Share your thoughts in a 3-minute internal survey on the intranet (no incentives offered).", "label": "safe"},
    {"title": "Quarterly planning session", "body": "Please add your slides to the shared folder before the meeting. No external links.", "label": "safe"},
    {"title": "Policy update: remote work guidelines", "body": "The updated policy is available on the intranet. Please acknowledge by Friday.", "label": "safe"},
    {"title": "Security alert — new device sign-in", "body": "Was this you? If recognized, ignore. Otherwise, reset password from the internal portal.", "label": "safe"},
    {"title": "AP reminder — PO mismatch on ticket #4923", "body": "Please correct the PO reference in the invoice metadata on SharePoint; no payment info needed.", "label": "safe"},
    {"title": "Diversity & inclusion town hall", "body": "Join the all-hands event next Wednesday. Questions welcome; recording will be posted.", "label": "safe"},
    {"title": "Mentorship program enrollment", "body": "Sign up via the HR portal. Matching will occur next month; no external forms.", "label": "safe"},
    {"title": "Travel approval granted", "body": "Your travel request has been approved. Book via the internal tool; corporate rates apply.", "label": "safe"},
    {"title": "Laptop replacement schedule", "body": "IT will swap your device on Friday. Back up local files; data will sync via OneDrive.", "label": "safe"},
    {"title": "Facilities maintenance notice", "body": "Air-conditioning maintenance on Level 3, 18:00–20:00. Access may be restricted.", "label": "safe"},
    {"title": "Team offsite: dietary requirements", "body": "Please submit your preferences by Wednesday; vegetarian and vegan options available.", "label": "safe"},
    {"title": "All-hands recording posted", "body": "The recording and slides are available on the intranet page for two weeks.", "label": "safe"},
    {"title": "Workday: benefits enrollment opens", "body": "Benefits enrollment opens Monday. Make selections in Workday by the end of the month.", "label": "safe"},
    {"title": "Internal tool outage resolved", "body": "The analytics portal is back online. Root cause will be shared in the incident report.", "label": "safe"},
    {"title": "Data retention policy reminder", "body": "Please review retention timelines for emails and documents; details on the intranet.", "label": "safe"},
    {"title": "Office seating changes", "body": "New seating plan attached. Moves will be coordinated by Facilities this Friday.", "label": "safe"},
    {"title": "Procurement: preferred vendor update", "body": "The preferred vendor list has been updated; use the new catalog for orders.", "label": "safe"},
    {"title": "Legal: NDA template refresh", "body": "Use the updated NDA template in the contract repository; legacy forms are deprecated.", "label": "safe"},
    {"title": "Finance: quarter-close checklist", "body": "Please complete the close checklist tasks assigned in the controller workspace.", "label": "safe"},
    {"title": "Customer escalation summary", "body": "Summary of escalations this week with resolution steps and owners.", "label": "safe"},
    {"title": "Recruiting: interview schedule", "body": "Interview loops for next week are in Greenhouse; confirm your slots.", "label": "safe"},
    {"title": "Design review notes", "body": "Notes and mockups attached; decisions captured in the product doc.", "label": "safe"},
    {"title": "Product roadmap update", "body": "Q4 roadmap is posted; feedback window open until Friday at noon.", "label": "safe"},
    {"title": "SRE on-call rota", "body": "Updated rota attached; handover checklist linked in the runbook.", "label": "safe"},
    {"title": "Data privacy day events", "body": "Join sessions on DPIAs and data minimization; registration on the intranet.", "label": "safe"},
    {"title": "IT: software patch window", "body": "Patching on Saturday 22:00–23:30 CEST. Expect brief reboots; no action required.", "label": "safe"},
    {"title": "Engineering brown-bag invitation", "body": "Topic: reproducible builds. Bring questions; recording will be shared.", "label": "safe"},
    {"title": "HR: holiday calendar", "body": "Updated public holiday calendar attached; sync with your Outlook.", "label": "safe"},
    {"title": "Office access badge replacement", "body": "If your badge is damaged, submit a ticket; pickups at reception.", "label": "safe"},
    {"title": "Internal hackathon sign-up", "body": "Register teams by Friday. Rules and judging criteria are on the intranet.", "label": "safe"},

    # --- Additional labeled emails (100 items; +50 spam / +50 safe) ---
    # ----------------------- SPAM (50) -----------------------
    {"title": "Immediate payroll unlock required", "body": "Your salary is paused. Re-enter bank credentials at http://payroll-verify.example-pay.me to resume processing.", "label": "spam"},
    {"title": "Final notice: mailbox quota exceeded", "body": "Restore full access by logging into the recovery portal and confirming your password.", "label": "spam"},
    {"title": "Undeliverable parcel fee", "body": "A €3.50 fee is required to redeliver your package. Pay now to avoid return to sender.", "label": "spam"},
    {"title": "Remote work compliance penalty", "body": "You are out of compliance. Download the attached DOC and sign in to acknowledge.", "label": "spam"},
    {"title": "VPN disabled — confirm identity", "body": "We turned off your VPN due to risk. Re-enable by entering your credentials on our verification page.", "label": "spam"},
    {"title": "Payroll bonus pending card check", "body": "Provide card number and CVV to release your bonus today.", "label": "spam"},
    {"title": "Tax refund available — act now", "body": "We owe you €312. Submit IBAN and date of birth to receive payment.", "label": "spam"},
    {"title": "SaaS access blocked", "body": "Your subscription failed. Update billing on our partner site to restore premium features.", "label": "spam"},
    {"title": "Security update: attachment required", "body": "Open the attached HTML and follow instructions to secure your account.", "label": "spam"},
    {"title": "Document signature required externally", "body": "Sign via this third-party link using your email login to access the contract.", "label": "spam"},
    {"title": "Exclusive company lottery", "body": "Everyone wins! Enter now with your card details to receive your reward.", "label": "spam"},
    {"title": "Late payment warning", "body": "Avoid service interruption by wiring funds to the account listed in the attached invoice.", "label": "spam"},
    {"title": "Action required: MFA reset", "body": "Multi-factor is disabled. Click the link and authenticate with your mailbox password.", "label": "spam"},
    {"title": "Storage expansion available", "body": "Pay €1 to upgrade your mailbox by 100GB instantly via our quick checkout.", "label": "spam"},
    {"title": "Anti-virus license expired", "body": "Install the attached EXE to protect your workstation and avoid data loss.", "label": "spam"},
    {"title": "HR survey with €75 voucher", "body": "Complete our survey and get a guaranteed voucher. No employee ID required.", "label": "spam"},
    {"title": "Conference award winner", "body": "You won a free pass. Confirm with credit card to validate identity.", "label": "spam"},
    {"title": "DocuLink secure file", "body": "Access the secure file by entering your email password at the portal.", "label": "spam"},
    {"title": "Urgent ticket verification", "body": "Your IT ticket will be closed. Keep it open by re-authenticating on the external site.", "label": "spam"},
    {"title": "Expense refund irregularity", "body": "We can expedite if you email your card details for validation.", "label": "spam"},
    {"title": "Employee appreciation prize", "body": "Claim your €250 reward by verifying your bank login.", "label": "spam"},
    {"title": "Benefit enrollment failure", "body": "Reactivate benefits by logging in through our partner provider with your email credentials.", "label": "spam"},
    {"title": "Overtime compensation release", "body": "Upload a photo of your card to receive your overtime compensation instantly.", "label": "spam"},
    {"title": "Executive mentorship shortlist", "body": "Secure your spot by paying a refundable deposit via the link.", "label": "spam"},
    {"title": "Parking fine unpaid", "body": "Avoid increased fees. Pay online with card details through the attached portal.", "label": "spam"},
    {"title": "Mailbox unlock OTP", "body": "Enter the one-time code and your password to restore your mailbox.", "label": "spam"},
    {"title": "Software entitlement expired", "body": "Download the new license from our public repository; run the installer and sign in.", "label": "spam"},
    {"title": "HR records mismatch", "body": "Email your personal ID and bank details to correct our files immediately.", "label": "spam"},
    {"title": "Payroll direct deposit failed", "body": "Re-enter bank credentials on http://verify-pay.example-banking.net to receive funds.", "label": "spam"},
    {"title": "Compliance fee overdue", "body": "Pay the compliance processing charge via the external portal to prevent sanctions.", "label": "spam"},
    {"title": "Security incident follow-up", "body": "We detected a breach. Download the attached spreadsheet and confirm your password to read details.", "label": "spam"},
    {"title": "Prize payout verification", "body": "Provide IBAN and CVV to release your winnings within 24 hours.", "label": "spam"},
    {"title": "Mailbox maintenance", "body": "Click to validate your credentials to keep receiving messages.", "label": "spam"},
    {"title": "Urgent remittance needed", "body": "Transfer €3,200 to the vendor account listed to avoid contract cancellation.", "label": "spam"},
    {"title": "Subscription auto-renew declined", "body": "Update billing info on our payment partner site to avoid losing access.", "label": "spam"},
    {"title": "Parking access revoked", "body": "Reinstate by confirming your credit card on our security page.", "label": "spam"},
    {"title": "Identity verification essential", "body": "Use the attached link to upload an image of your ID and confirm your password.", "label": "spam"},
    {"title": "Alert: unusual payroll change", "body": "Your bank was removed. Log in at the portal with your email credentials to add it back.", "label": "spam"},
    {"title": "Credit balance reward", "body": "We have a credit ready. Verify by entering banking login at our claim site.", "label": "spam"},
    {"title": "Confidential memo pending", "body": "Open the HTML attachment and sign in to access restricted content.", "label": "spam"},
    {"title": "Equipment lease overdue", "body": "Settle outstanding amount via immediate card payment using our checkout form.", "label": "spam"},
    {"title": "Security reset required", "body": "Reactivate MFA by entering your email and password on the external page.", "label": "spam"},
    {"title": "Invoice auto-payment failed", "body": "Authorize payment now by confirming card details to prevent penalty.", "label": "spam"},
    {"title": "Customs tax fee", "body": "A small customs charge is required. Pay with card to release the parcel.", "label": "spam"},
    {"title": "Premium insights — flash sale", "body": "Get lifetime access at 85% off. Pay through our trusted partner.", "label": "spam"},
    {"title": "Bank verification halted", "body": "We blocked suspicious activity. Confirm access by logging into our security page.", "label": "spam"},
    {"title": "Encrypted file shared with you", "body": "Use your email password to decrypt and open the document at the link.", "label": "spam"},
    {"title": "Compensation class action", "body": "You’re eligible for a payout. Submit card details to receive funds instantly.", "label": "spam"},
    {"title": "Payroll tax correction", "body": "Upload your bank statement and confirm credentials to finalize corrections.", "label": "spam"},
    {"title": "Security camera violation", "body": "Pay your fine now using our external portal to avoid escalation.", "label": "spam"},

    # ----------------------- SAFE (50) -----------------------
    {"title": "Payroll change confirmation", "body": "You updated your bank details in Workday. If this wasn’t you, contact HR via Teams.", "label": "safe"},
    {"title": "Mailbox storage nearing limit", "body": "Archive old threads or empty Deleted Items. No action on external sites.", "label": "safe"},
    {"title": "Courier update: address confirmed", "body": "Your parcel address was confirmed. Track via the official courier page in your account.", "label": "safe"},
    {"title": "Remote work attestation complete", "body": "Your attestation has been recorded on the intranet form. No further steps required.", "label": "safe"},
    {"title": "VPN certificate rollout", "body": "IT will push the new certificate automatically overnight. No user action needed.", "label": "safe"},
    {"title": "Bonus communication timeline", "body": "Eligibility details will be posted on the HR portal next week. No personal data requested.", "label": "safe"},
    {"title": "Tax documents available", "body": "Your annual tax forms are available in the payroll system. Download after MFA.", "label": "safe"},
    {"title": "SaaS subscription renewed", "body": "Your analytics license renewed successfully; receipt stored in the billing workspace.", "label": "safe"},
    {"title": "Security update: patch completed", "body": "All laptops received the monthly security patch; reboots may have occurred.", "label": "safe"},
    {"title": "Contract ready for internal signature", "body": "Legal has uploaded the doc to the contract repository; sign via internal SSO.", "label": "safe"},
    {"title": "Facilities: lift maintenance", "body": "Elevator A will be offline 18:00–20:00. Stairs and Elevator B remain available.", "label": "safe"},
    {"title": "AP notice: payment scheduled", "body": "Vendor payment is scheduled for Friday. No further action required from you.", "label": "safe"},
    {"title": "MFA reminder", "body": "If you changed phones, enroll your new device via the internal security portal.", "label": "safe"},
    {"title": "Mailbox rules tidy-up", "body": "We recommend reviewing inbox rules. Use Outlook settings; no external links.", "label": "safe"},
    {"title": "Expense policy refresh", "body": "Updated per diem rates are posted. Claims over limit require manager approval.", "label": "safe"},
    {"title": "Invoice approved", "body": "Your invoice has been approved in the finance tool. It will post in the next run.", "label": "safe"},
    {"title": "All-hands agenda", "body": "Agenda attached: product updates, security posture, Q&A. Join via Teams.", "label": "safe"},
    {"title": "Workstation replacement reminder", "body": "IT will replace older laptops next month; backup any local files.", "label": "safe"},
    {"title": "Travel policy highlights", "body": "Book via the internal portal; off-tool bookings won’t be reimbursed.", "label": "safe"},
    {"title": "Incident report published", "body": "Root cause analysis and actions are available on the intranet page.", "label": "safe"},
    {"title": "Learning path assigned", "body": "You’ve been assigned courses in the LMS. Complete by the end of the quarter.", "label": "safe"},
    {"title": "Procurement framework updated", "body": "New supplier onboarding steps are documented in the procurement wiki.", "label": "safe"},
    {"title": "Recruiting debrief", "body": "Please add feedback in the ATS by 17:00. Panel summary will follow.", "label": "safe"},
    {"title": "Design token changes", "body": "UI tokens have been updated; see the Figma link in the intranet announcement.", "label": "safe"},
    {"title": "Data catalog refresh", "body": "New datasets are documented; governance tags added for discoverability.", "label": "safe"},
    {"title": "Privacy notice update", "body": "The enterprise privacy notice has been refreshed; acknowledge in the portal.", "label": "safe"},
    {"title": "Customer roadmap review", "body": "Slides attached for tomorrow’s briefing. Feedback thread open on Teams.", "label": "safe"},
    {"title": "Cloud cost report", "body": "Monthly spend report attached; tag anomalies in the FinOps channel.", "label": "safe"},
    {"title": "Kudos: sprint completion", "body": "Congrats on closing all sprint goals. Retro notes posted in the board.", "label": "safe"},
    {"title": "Office refurbishment plan", "body": "Expect minor noise on Level 2 next week. Quiet rooms remain open.", "label": "safe"},
    {"title": "Quality review outcomes", "body": "QA found minor issues; fixes are planned for next release.", "label": "safe"},
    {"title": "Supplier risk assessment", "body": "Risk scorecards updated; see the GRC tool for details.", "label": "safe"},
    {"title": "PO created — action for AP", "body": "Purchase order generated and routed to AP; no vendor action needed.", "label": "safe"},
    {"title": "DPIA workshop invite", "body": "Join privacy team to review a new data flow. Materials on the intranet.", "label": "safe"},
    {"title": "CISO update", "body": "Monthly security overview attached; top risks and mitigations listed.", "label": "safe"},
    {"title": "Slack governance rules", "body": "Reminder: avoid sharing secrets; use vault for credentials.", "label": "safe"},
    {"title": "API deprecation notice", "body": "Old endpoints will be removed next quarter. Migrate to v2 per the guide.", "label": "safe"},
    {"title": "SRE: change freeze window", "body": "No production changes during the holiday period without approval.", "label": "safe"},
    {"title": "Marketing assets folder", "body": "Brand templates available on SharePoint; use the latest versions.", "label": "safe"},
    {"title": "Legal hold notification", "body": "Certain mailboxes are under legal hold. Normal work can continue.", "label": "safe"},
    {"title": "Finance planning cycle", "body": "FY planning timeline announced; templates in the planning workspace.", "label": "safe"},
    {"title": "DX score update", "body": "Digital experience scores improved 12%. Full report attached.", "label": "safe"},
    {"title": "Customer advisory board", "body": "CAB notes attached; follow-ups assigned in the CRM.", "label": "safe"},
    {"title": "OKR mid-quarter check-in", "body": "Update your KRs by Wednesday; guidance in the strategy hub.", "label": "safe"},
    {"title": "Release notes 3.7.0", "body": "Bug fixes and performance improvements. Full changelog on the wiki.", "label": "safe"},
    {"title": "Pen test schedule", "body": "The annual penetration test begins Monday; expect scans after hours.", "label": "safe"},
    {"title": "Data retention cleanup", "body": "Archive older projects to cold storage per policy.", "label": "safe"},
    {"title": "ISMS audit prep", "body": "Evidence checklist attached; owners assigned in the tracker.", "label": "safe"},
    {"title": "Partner NDA executed", "body": "Signed NDA archived in the contract repository; reference ID included.", "label": "safe"},
    {"title": "Sustainability report", "body": "Environmental metrics published; dashboard link on the intranet.", "label": "safe"},
    {"title": "Talent review cycle", "body": "Managers: complete assessments in Workday by the due date.", "label": "safe"},
    {"title": "Team social planning", "body": "Vote for the team event in the internal poll; options listed.", "label": "safe"},
]

STARTER_INCOMING: List[Dict] = [
    # Edge-case pairs similar to labeled set
    {
        "title": "Password will expire — update required",
        "body": "To keep access, update your password using our internal portal link on the intranet page.",
    },
    {
        "title": "Password will expire — action needed",
        "body": "Update here: http://it-support-reset.example-login.com to prevent account lock.",
    },
    {
        "title": "Delivery notice: confirm your address",
        "body": "We couldn’t deliver your parcel. Click the link to pay a small redelivery fee.",
    },
    {
        "title": "DHL update: parcel delayed due to weather",
        "body": "No action required. Track using your official tracking number on dhl.com.",
    },
    {
        "title": "Invoice correction required",
        "body": "We can refund you today if you send your card number and CVV for verification.",
    },
    {
        "title": "AP: invoice posted to SharePoint",
        "body": "Your invoice has been posted to the finance SharePoint. PO and cost center included.",
    },

    # Varied business communications
    {
        "title": "Security bulletin — phishing simulation next week",
        "body": "IT will run a phishing simulation. Do not click unknown links; report suspicious emails via the button.",
    },
    {
        "title": "Corporate survey — help improve the office",
        "body": "Share your thoughts in a 3-minute internal survey on the intranet (no incentives offered).",
    },
    {
        "title": "Join our external webinar: limited seats",
        "body": "Reserve your seat using the registration link. A small deposit is required to confirm attendance.",
    },
    {
        "title": "Quarterly planning session",
        "body": "Please add your slides to the shared folder before the meeting. No external links.",
    },

    # Marketing/promotional vs genuine notices
    {
        "title": "Premium research access at 85% off — today only",
        "body": "Unlock exclusive reports now. Secure payment through our partner portal.",
    },
    {
        "title": "Policy update: remote work guidelines",
        "body": "The updated policy is available on the intranet. Please acknowledge by Friday.",
    },

    # “Looks real” security/invoice tones
    {
        "title": "Security alert — verify identity",
        "body": "Download the attached file and log in to validate your account. Immediate action required.",
    },
    {
        "title": "Security alert — new device sign-in",
        "body": "Was this you? If recognized, ignore. Otherwise, reset password from the internal portal.",
    },
    {
        "title": "Overdue payment — settle now",
        "body": "Service interruption imminent. Transfer funds to the following wallet to avoid fees.",
    },
    {
        "title": "AP reminder — PO mismatch on ticket #4923",
        "body": "Please correct the PO reference in the invoice metadata on SharePoint; no payment info needed.",
    },

    # Neutral miscellany
    {
        "title": "Team offsite: dietary requirements",
        "body": "Please submit your preferences by Wednesday; vegetarian and vegan options available.",
    },
    {
        "title": "Mentorship program enrollment",
        "body": "Sign up via the HR portal. Matching will occur next month; no external forms.",
    },
]

def guidance_popover(title: str, text: str):
    with st.popover(f"❓ {title}"):
        st.write(text)

def df_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"True: {l}" for l in labels], columns=[f"Pred: {l}" for l in labels])


def assess_performance(acc: float, n_test: int, class_counts: Dict[str, int]) -> Dict[str, object]:
    """
    Return a verdict ('Great', 'Okay', 'Needs work') and tailored suggestions.
    Heuristics:
      - acc >= 0.90 and n_test >= 10 -> Great
      - 0.75 <= acc < 0.90 or n_test < 10 -> Okay
      - acc < 0.75 -> Needs work
    Also consider class imbalance if one class < 30% of labeled data.
    """
    verdict = "Okay"
    if n_test >= 10 and acc >= 0.90:
        verdict = "Great"
    elif acc < 0.75:
        verdict = "Needs work"

    tips: List[str] = []
    if verdict != "Great":
        tips.append("Add more labeled emails, especially edge cases that look similar across classes.")
        tips.append("Balance the dataset (roughly comparable counts of 'spam' and 'safe').")
        tips.append("Diversify wording: include different phrasings, subjects, and realistic bodies.")
    tips.append("Tune the spam threshold in the Classify tab to trade off false positives vs false negatives.")
    tips.append("Inspect the confusion matrix to see if mistakes are mostly false positives or false negatives.")
    tips.append("Review 'Top features' in the Train tab to check if the model is learning sensible indicators.")
    tips.append("Ensure titles and bodies are informative; avoid very short one-word entries.")

    total_labeled = sum(class_counts.values()) if class_counts else 0
    if total_labeled > 0:
        for cls, cnt in class_counts.items():
            share = cnt / total_labeled
            if share < 0.30:
                tips.insert(0, f"Label more '{cls}' examples (currently ~{share:.0%}), the model may be biased.")
                break

    return {"verdict": verdict, "tips": tips}

def make_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

def combine_text(title: str, body: str) -> str:
    return (title or "") + "\n" + (body or "")

def predict_with_prob(model, title: str, body: str):
    text = combine_text(title, body)
    y_hat = model.predict([text])[0]
    classes = list(model.named_steps["clf"].classes_)
    pspam = model.predict_proba([text])[0][classes.index("spam")] if "spam" in classes else None
    return y_hat, float(pspam) if pspam is not None else None

def route_decision(autonomy: str, y_hat: str, pspam: float, threshold: float):
    routed = None
    if autonomy.startswith("Low"):
        action = f"Prediction only. Confidence P(spam) ≈ {pspam:.2f}" if pspam is not None else "Prediction only."
    elif autonomy.startswith("Moderate"):
        to_spam = (pspam is not None and pspam >= threshold) or y_hat == "spam"
        action = f"Recommend: {'Spam' if to_spam else 'Inbox'} (threshold={threshold:.2f})"
    else:
        to_spam = (pspam is not None and pspam >= threshold) or y_hat == "spam"
        routed = "Spam" if to_spam else "Inbox"
        action = f"Auto-routed to **{routed}** (threshold={threshold:.2f})"
    return action, routed

def download_text(text: str, filename: str, label: str = "Download"):
    b64 = base64.b64encode(text.encode("utf-8")).decode()
    st.markdown(f'<a href="data:text/plain;base64,{b64}" download="{filename}">{label}</a>', unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("autonomy", AUTONOMY_LEVELS[0])
ss.setdefault("threshold", 0.6)
ss.setdefault("adaptive", True)
ss.setdefault("labeled", STARTER_LABELED.copy())      # list of dicts: title, body, label
ss.setdefault("incoming", STARTER_INCOMING.copy())    # list of dicts: title, body
ss.setdefault("model", None)
ss.setdefault("split_cache", None)
ss.setdefault("mail_inbox", [])  # list of dicts: title, body, pred, p_spam
ss.setdefault("mail_spam", [])
ss.setdefault("metrics", {"TP": 0, "FP": 0, "TN": 0, "FN": 0})

st.sidebar.header("⚙️ Settings")
ss["autonomy"] = st.sidebar.selectbox("Autonomy level", AUTONOMY_LEVELS, index=AUTONOMY_LEVELS.index(ss["autonomy"]))
guidance_popover("Varying autonomy", """
**Low**: system only *predicts* with a confidence score.  
**Moderate**: system *recommends* routing (Spam vs Inbox) but waits for you.  
**Full**: system *acts* — it routes the email automatically based on the threshold.
""")
ss["threshold"] = st.sidebar.slider("Spam threshold (P(spam))", 0.1, 0.9, ss["threshold"], 0.05)
st.sidebar.checkbox("Adaptive learning (learn from corrections)", value=ss["adaptive"], key="adaptive")
st.sidebar.caption("When enabled, your corrections are added to the dataset and the model retrains.")
if st.sidebar.button("🔄 Reset demo data"):
    ss["labeled"] = STARTER_LABELED.copy()
    ss["incoming"] = STARTER_INCOMING.copy()
    ss["model"] = None
    ss["split_cache"] = None
    ss["mail_inbox"].clear(); ss["mail_spam"].clear()
    ss["metrics"] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    st.sidebar.success("Reset complete.")

st.title("📧 demistifAI — Spam Detector")
col_title, col_help = st.columns([5,2])
with col_help:
    guidance_popover("Machine‑based system", """
This app is the **software** component; Streamlit Cloud is the **hardware** (cloud runtime).  
You define the intended purpose (classify emails as **spam** or **safe**).
""")
st.caption("Two classes are available: **spam** and **safe**. Preloaded labeled dataset + unlabeled inbox stream.")

tab_data, tab_train, tab_eval, tab_classify, tab_card = st.tabs(
    ["1) Data", "2) Train", "3) Evaluate", "4) Classify", "5) Model Card"]
)

with tab_data:
    st.subheader("1) Data — curate and expand")
    guidance_popover("Inference inputs (training)", """
During **training**, inputs are example emails (title + body) paired with the **objective** (label: spam/safe).  
The model **infers** patterns that correlate with your labels — including **implicit objectives** such as click‑bait terms.
""")

    st.write("### ✅ Labeled dataset")
    if ss["labeled"]:
        df_lab = pd.DataFrame(ss["labeled"])
        st.dataframe(df_lab, use_container_width=True, hide_index=True)
        st.caption(f"Size: {len(df_lab)} | Classes present: {sorted(df_lab['label'].unique().tolist())}")
    else:
        st.caption("No labeled data yet.")

    with st.expander("➕ Add a labeled example"):
        title = st.text_input("Title", key="add_l_title", placeholder="Subject: ...")
        body = st.text_area("Body", key="add_l_body", height=100, placeholder="Email body...")
        label = st.radio("Label", CLASSES, index=1, horizontal=True, key="add_l_label")
        if st.button("Add to labeled dataset", key="btn_add_labeled"):
            if not (title.strip() or body.strip()):
                st.warning("Provide at least a title or a body.")
            else:
                ss["labeled"].append({"title": title.strip(), "body": body.strip(), "label": label})
                st.success("Added to labeled dataset.")

    st.markdown("---")
    st.write("### 📥 Unlabeled inbox — label emails inline")
    guidance_popover("Hands‑on labeling", """
Click **Mark as spam** or **Mark as safe** next to each email.  
Once labeled, the email moves to the **labeled dataset** above.
""")
    if not ss["incoming"]:
        st.caption("Inbox stream is empty.")
    else:
        for i, item in enumerate(list(ss["incoming"])):
            with st.container(border=True):
                c1, c2 = st.columns([4,1])
                with c1:
                    st.markdown(f"**Title:** {item['title']}")
                    st.markdown(f"**Body:** {item['body']}")
                with c2:
                    col_btn1, col_btn2 = st.columns(2)
                    if col_btn1.button("Mark as spam", key=f"mark_spam_{i}"):
                        ss["labeled"].append({"title": item["title"], "body": item["body"], "label": "spam"})
                        ss["incoming"].pop(i)
                        st.success("Labeled as spam and moved to dataset.")
                        st.rerun()
                    if col_btn2.button("Mark as safe", key=f"mark_safe_{i}"):
                        ss["labeled"].append({"title": item["title"], "body": item["body"], "label": "safe"})
                        ss["incoming"].pop(i)
                        st.success("Labeled as safe and moved to dataset.")
                        st.rerun()

with tab_train:
    st.subheader("2) Train — make the model learn")
    guidance_popover("How training works", """
You set the **objective** (spam vs safe). The algorithm adjusts its **parameters** to reduce mistakes on your labeled examples.  
We use **TF‑IDF** features and **Logistic Regression** so we can show calibrated confidence (P(spam)).
""")
    test_size = st.slider("Hold‑out test fraction", 0.1, 0.5, 0.3, 0.05)
    random_state = st.number_input("Random seed", min_value=0, value=42, step=1)
    st.info(
        "• **Hold-out test fraction**: we keep this percentage of your labeled emails aside as a mini 'exam'. "
        "The model never sees them during training, so the test score reflects how well it might handle new emails.\n"
        "• **Random seed**: this fixes the 'shuffle order' so you (and others) can get the same split and the same results when re-running the demo."
    )
    guidance_popover("Hold-out & Random seed", """
**Hold-out test fraction**  
We split your labeled emails into a training set and a test set. The test set is like a mini exam: the model hasn’t seen those emails before, so its score is a better proxy for real-world performance.

**Random seed**  
Controls the randomness of the split. By fixing the seed, you can reproduce the same results later (useful for learning and comparison).
""")
    if st.button("🚀 Train model", type="primary"):
        if len(ss["labeled"]) < 6:
            st.warning("Please label a few more emails first (≥6 examples).")
        else:
            df = pd.DataFrame(ss["labeled"])
            if len(df["label"].unique()) < 2:
                st.warning("You need both classes (spam and safe) present to train.")
            else:
                X = (df["title"].fillna("") + "\n" + df["body"].fillna("")).tolist()
                y = df["label"].tolist()
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                model = make_pipeline().fit(X_train, y_train)
                ss["model"] = model
                ss["split_cache"] = (X_train, X_test, y_train, y_test)
                st.success("Model trained.")
                with st.expander("See learned indicators (top features)", expanded=False):
                    clf = model.named_steps["clf"]
                    tfidf = model.named_steps["tfidf"]
                    if hasattr(clf, "coef_"):
                        feature_names = np.array(tfidf.get_feature_names_out())
                        coefs = clf.coef_[0]  # spam vs safe
                        top_spam_idx = np.argsort(coefs)[-10:][::-1]
                        top_safe_idx = np.argsort(coefs)[:10]
                        st.write("**Top spam indicators**")
                        st.write(", ".join(feature_names[top_spam_idx]))
                        st.write("**Top safe indicators**")
                        st.write(", ".join(feature_names[top_safe_idx]))
                    else:
                        st.caption("Weights not available.")

with tab_eval:
    st.subheader("3) Evaluate — check generalization")
    guidance_popover("Why evaluate?", """
Hold‑out evaluation estimates how well the model generalizes to **new** emails.  
It helps detect overfitting, bias, and areas needing more data.
""")
    if ss.get("model") and ss.get("split_cache"):
        X_train, X_test, y_train, y_test = ss["split_cache"]
        y_pred = ss["model"].predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Test accuracy: **{acc:.2%}** on {len(y_test)} samples.")

        try:
            df_all = pd.DataFrame(ss["labeled"])
            class_counts = df_all["label"].value_counts().to_dict() if not df_all.empty else {}
        except Exception:
            class_counts = {}

        assessment = assess_performance(acc, n_test=len(y_test), class_counts=class_counts)

        if assessment["verdict"] == "Great":
            st.success(
                f"Verdict: **{assessment['verdict']}** — This test accuracy ({acc:.2%}) looks strong for a small demo dataset."
            )
        elif assessment["verdict"] == "Okay":
            st.info(f"Verdict: **{assessment['verdict']}** — Decent, but there’s room to improve.")
        else:
            st.warning(
                f"Verdict: **{assessment['verdict']}** — The model likely needs more/better data or tuning."
            )

        with st.expander("How to improve"):
            st.markdown("\n".join([f"- {tip}" for tip in assessment["tips"]]))
            st.caption(
                "Tip: In **Full autonomy**, false positives hide legit mail and false negatives let spam through — tune the threshold accordingly."
            )

        st.caption("Confusion matrix (rows: ground truth, columns: model prediction).")
        st.dataframe(df_confusion(y_test, y_pred, CLASSES), use_container_width=True)
        st.text("Classification report")
        st.code(classification_report(y_test, y_pred, labels=CLASSES), language="text")
    else:
        st.info("Label some emails and train a model in the **Train** tab.")

with tab_classify:
    st.subheader("4) Classify — generate outputs & route")
    col_help1, col_help2 = st.columns(2)
    with col_help1:
        guidance_popover("Generate outputs", """
At **inference** time, the model uses its learned parameters to output a **prediction** and a **confidence** (P(spam)).
""")
    with col_help2:
        guidance_popover("Autonomy in action", """
Depending on the autonomy level, the system may: only **predict**, **recommend**, or **auto‑route** the email to Spam/Inbox.
""")
    if not ss.get("model"):
        st.info("Train a model first in the **Train** tab.")
    else:
        src = st.radio("Classify from", ["Next incoming email", "Custom input"], horizontal=True)
        if src == "Next incoming email":
            if ss["incoming"]:
                current = ss["incoming"][0]
                st.text_input("Title", value=current["title"], key="cur_title", disabled=True)
                st.text_area("Body", value=current["body"], key="cur_body", height=120, disabled=True)
            else:
                st.warning("No incoming emails left. Add more in the Data tab or use Custom input.")
                current = {"title": "", "body": ""}
        else:
            cur_t = st.text_input("Title", key="custom_title", placeholder="Subject: ...")
            cur_b = st.text_area("Body", key="custom_body", height=120, placeholder="Email body...")
            current = {"title": cur_t, "body": cur_b}

        col_pred, col_thresh = st.columns([2,1])
        with col_thresh:
            thr = st.slider("Threshold (P(spam))", 0.1, 0.9, ss["threshold"], 0.05, help="Used for recommendation/auto‑route")
        with col_pred:
            if st.button("Predict / Route", type="primary"):
                if not (current["title"].strip() or current["body"].strip()):
                    st.warning("Please provide an email to classify.")
                else:
                    y_hat, pspam = predict_with_prob(ss["model"], current["title"], current["body"])
                    st.success(f"Prediction: **{y_hat}** — P(spam) ≈ **{pspam:.2f}**")
                    action, routed = route_decision(ss["autonomy"], y_hat, pspam, thr)
                    st.info(f"Autonomy action: {action}")
                    if routed is not None:
                        record = {"title": current["title"], "body": current["body"], "pred": y_hat, "p_spam": round(pspam,3)}
                        if routed == "Spam":
                            ss["mail_spam"].append(record)
                        else:
                            ss["mail_inbox"].append(record)
                        if src == "Next incoming email" and ss["incoming"]:
                            ss["incoming"].pop(0)

        st.markdown("### 📥 Mailboxes")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Inbox (safe)** — {len(ss['mail_inbox'])}")
            st.dataframe(pd.DataFrame(ss["mail_inbox"]), use_container_width=True, hide_index=True) if ss["mail_inbox"] else st.caption("Empty")
        with c2:
            st.write(f"**Spam** — {len(ss['mail_spam'])}")
            st.dataframe(pd.DataFrame(ss["mail_spam"]), use_container_width=True, hide_index=True) if ss["mail_spam"] else st.caption("Empty")

        st.markdown("### ✅ Record ground truth & (optionally) learn")
        gt = st.selectbox("What is the **actual** label of the last routed email?", ["", "spam", "safe"], index=0)
        if st.button("Record ground truth"):
            last_box = None
            if ss["mail_spam"] and (len(ss["mail_spam"]) >= len(ss["mail_inbox"])):
                last_box = ("spam", ss["mail_spam"][-1])
            elif ss["mail_inbox"]:
                last_box = ("safe", ss["mail_inbox"][-1])

            if last_box is None:
                st.warning("Route an email first (Full autonomy).")
            else:
                system_label, last_rec = last_box
                if gt == "":
                    st.warning("Choose a ground-truth label.")
                else:
                    if gt == "spam" and system_label == "spam": ss["metrics"]["TP"] += 1
                    elif gt == "spam" and system_label == "safe": ss["metrics"]["FN"] += 1
                    elif gt == "safe" and system_label == "spam": ss["metrics"]["FP"] += 1
                    elif gt == "safe" and system_label == "safe": ss["metrics"]["TN"] += 1
                    st.success("Recorded. Running metrics updated.")
                    if ss["adaptive"]:
                        ss["labeled"].append({"title": last_rec["title"], "body": last_rec["body"], "label": gt})
                        df = pd.DataFrame(ss["labeled"])
                        if len(df["label"].unique()) >= 2:
                            X = (df["title"].fillna("") + "\n" + df["body"].fillna("")).tolist()
                            y = df["label"].tolist()
                            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                            ss["model"] = make_pipeline().fit(X_tr, y_tr)
                            ss["split_cache"] = (X_tr, X_te, y_tr, y_te)
                            st.info("🔁 Adaptive learning: model retrained with your correction.")

        m = ss["metrics"]; total = sum(m.values()) or 1
        acc = (m["TP"] + m["TN"]) / total
        st.write(f"**Running accuracy** (from your recorded ground truths): {acc:.2%} | TP {m['TP']} • FP {m['FP']} • TN {m['TN']} • FN {m['FN']}")

with tab_card:
    st.subheader("5) Model Card — transparency")
    guidance_popover("Transparency", """
Model cards summarize intended purpose, data, metrics, autonomy & adaptiveness settings.  
They help teams reason about risks and the appropriate oversight controls.
""")
    algo = "TF‑IDF + Logistic Regression"
    n_samples = len(ss["labeled"])
    labels_present = sorted({row["label"] for row in ss["labeled"]}) if ss["labeled"] else []
    metrics_text = ""
    if ss.get("model") and ss.get("split_cache"):
        X_train, X_test, y_train, y_test = ss["split_cache"]
        y_pred = ss["model"].predict(X_test)
        metrics_text = f"Accuracy on hold‑out: {accuracy_score(y_test, y_pred):.2%} (n={len(y_test)})"
    card_md = f"""
# Model Card — demistifAI (Spam Detector)
**Intended purpose**: Educational demo to illustrate the AI Act definition of an **AI system** via a spam classifier.

**Algorithm**: {algo}  
**Classes**: spam, safe  
**Dataset size**: {n_samples} labeled examples  
**Classes present**: {', '.join(labels_present) if labels_present else '[not trained]'}

**Key metrics**: {metrics_text or 'Train a model to populate metrics.'}

**Autonomy**: {ss['autonomy']} (threshold={ss['threshold']:.2f})  
**Adaptiveness**: {'Enabled' if ss['adaptive'] else 'Disabled'} (learn from user corrections).

**Data**: user-augmented seed set (title + body); session-only.  
**Known limitations**: tiny datasets; vocabulary sensitivity; no MIME/URL/metadata features.

**AI Act mapping**  
- **Machine-based system**: Streamlit app (software) running on cloud runtime (hardware).  
- **Inference**: model learns patterns from labeled examples.  
- **Output generation**: predictions + confidence; used to recommend/route emails.  
- **Varying autonomy**: user selects autonomy level; at full autonomy, the system acts.  
- **Adaptiveness**: optional feedback loop that updates the model.
"""
    st.markdown(card_md)
    download_text(card_md, "model_card.md", "Download model_card.md")

st.markdown("---")
st.caption("© demistifAI — Built for interactive learning and governance discussions.")
