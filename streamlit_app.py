import base64
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="demistifAI — Spam Detector", page_icon="📧", layout="wide")

CLASSES = ["spam", "safe"]
AUTONOMY_LEVELS = [
    "Low autonomy (predict + confidence)",
    "Moderate autonomy (recommendation)",
    "Full autonomy (auto-route)",
]

STARTER_LABELED: List[Dict] = [
    # ----------------------- SPAM (100) -----------------------
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

    # ----------------------- SAFE (100) -----------------------
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
    {"title": "Payroll card verification needed", "body": "We couldn't process your salary card. Submit the 3D Secure code through our payment portal within 30 minutes.", "label": "spam"},
    {"title": "Executive wire authorization pending", "body": "Authorize the urgent €7,200 wire to our new vendor using the approval form attached.", "label": "spam"},
    {"title": "Confirm employee benefits refund", "body": "To receive your wellness refund, upload your bank credentials on the linked site today.", "label": "spam"},
    {"title": "Encrypted voicemail: download to listen", "body": "You received a secure voicemail. Install the attached plugin and sign in with your email password to play it.", "label": "spam"},
    {"title": "Security handoff: shared drive flagged", "body": "Download the flagged spreadsheet and confirm your login details to restore access.", "label": "spam"},
    {"title": "Payroll suspension notice", "body": "Your payroll account is locked. Verify your identity by entering your SSN and password at hr-verify.co.", "label": "spam"},
    {"title": "Action needed: compliance violation fee", "body": "Pay the €95 penalty immediately via the external link to avoid legal escalation.", "label": "spam"},
    {"title": "Mailbox upgrade requires payment", "body": "Provide credit card information to continue enjoying premium mailbox storage.", "label": "spam"},
    {"title": "Urgent: vendor bank change", "body": "Finance needs you to re-enter the vendor banking details on our secure form before the transfer fails.", "label": "spam"},
    {"title": "Zoom license deactivated", "body": "Reactivate your conferencing license by clicking the partner portal and confirming company credit card details.", "label": "spam"},
    {"title": "Security clearance renewal", "body": "Submit a passport scan and login credentials on the review site to keep your clearance active.", "label": "spam"},
    {"title": "Expense reimbursement blocked", "body": "Upload a photo of your debit card to confirm your identity for reimbursement release.", "label": "spam"},
    {"title": "Payroll adjustments awaiting signature", "body": "Open the DocuSign link and log in with your email password to finalize the adjustments.", "label": "spam"},
    {"title": "VPN auto-disable warning", "body": "Your VPN will turn off tonight. Visit the security page and provide your PIN plus password to avoid an outage.", "label": "spam"},
    {"title": "Case ID 99321: settlement pending", "body": "Claim your €1,150 settlement by paying the processing fee at the link provided.", "label": "spam"},
    {"title": "Apple Pay card removal", "body": "We removed your card. Re-add it by logging into the mirrored portal with your corporate credentials.", "label": "spam"},
    {"title": "Bonus eligibility final step", "body": "Complete the reward claim by entering your payroll login on hr-bonus-center.com.", "label": "spam"},
    {"title": "MFA reset confirmation", "body": "Ignore other notices. Use this link to disable MFA and verify with your password now.", "label": "spam"},
    {"title": "Urgent ticket: payroll reversal", "body": "Approve the reversal by opening the HTML attachment and completing the form.", "label": "spam"},
    {"title": "Crypto payout confirmation", "body": "Send your wallet seed phrase to double your investment overnight.", "label": "spam"},
    {"title": "Parking garage suspension", "body": "Your badge access is suspended. Pay the reinstatement fee via the quickpay form with card details.", "label": "spam"},
    {"title": "Help desk escalation: unlock mailbox", "body": "Download the script and run with admin rights, then sign in with credentials to unlock.", "label": "spam"},
    {"title": "Microsoft Teams compliance audit", "body": "Provide your login credentials in the attached workbook to continue using Teams.", "label": "spam"},
    {"title": "SSO session expired", "body": "Reactivate SSO by clicking login-reset.io and confirming your password plus SMS code.", "label": "spam"},
    {"title": "E-signature failure", "body": "The document failed to sign. Authenticate at doc-fix.me with your company login to resend.", "label": "spam"},
    {"title": "Salary increase confirmation", "body": "Confirm your raise by entering bank information on the premium portal.", "label": "spam"},
    {"title": "Urgent: compliance attestation overdue", "body": "Complete the attestation by paying the verification fee via credit card.", "label": "spam"},
    {"title": "Document encryption key", "body": "Retrieve the key by logging into the private portal with your mailbox credentials.", "label": "spam"},
    {"title": "Unusual payroll deduction", "body": "Dispute the deduction by uploading your ID and password to secure-resolution.app.", "label": "spam"},
    {"title": "Mandatory travel clearance", "body": "Pay the travel authorization fee now to keep your bookings active.", "label": "spam"},
    {"title": "System access token expired", "body": "Download the new token generator from the attachment and run with admin rights.", "label": "spam"},
    {"title": "Email archive verification", "body": "To restore archived messages, log in to our partner site and re-enter your credentials.", "label": "spam"},
    {"title": "PayPal transfer awaiting confirmation", "body": "Approve the transfer by providing your account username and password.", "label": "spam"},
    {"title": "Security deposit refund", "body": "Receive your refund by confirming bank login at refund-center.link.", "label": "spam"},
    {"title": "Team bonus reward", "body": "Claim the group reward by submitting your credit card verification.", "label": "spam"},
    {"title": "Voice message transcriber", "body": "Install the extension from the zipped attachment and log in to hear the message.", "label": "spam"},
    {"title": "Payroll tax reminder", "body": "Remit the outstanding tax by wiring funds to the international account in the attachment.", "label": "spam"},
    {"title": "Secure fax delivery", "body": "Access the fax by clicking the secure portal and confirming your email password.", "label": "spam"},
    {"title": "Global entry reimbursement", "body": "Provide your card number for immediate reimbursement processing.", "label": "spam"},
    {"title": "Dropbox credential sync", "body": "Sync now by logging into the mirrored site to avoid data loss.", "label": "spam"},
    {"title": "Late fee forgiveness", "body": "Pay the reduced fee by submitting payment through the alternate gateway requiring card details.", "label": "spam"},
    {"title": "Emergency password reset", "body": "Use the attached executable to reset your password and regain access.", "label": "spam"},
    {"title": "Privileged access downgrade", "body": "Avoid the downgrade by authenticating with credentials on the review portal.", "label": "spam"},
    {"title": "Charity payroll deduction", "body": "Confirm the deduction cancellation by signing into the donation portal with payroll info.", "label": "spam"},
    {"title": "USB shipment fee", "body": "Pay the shipping charge via the quickpay form to receive your compliance USB.", "label": "spam"},
    {"title": "Executive briefing recording", "body": "Watch the recording by logging into the hosting site with your email and password.", "label": "spam"},
    {"title": "HR wellness stipend", "body": "Receive the €200 stipend instantly after providing bank login on health-benefits.io.", "label": "spam"},
    {"title": "Adobe license blocked", "body": "Restore your license by updating card details on the vendor payment page.", "label": "spam"},
    {"title": "Overtime payout validation", "body": "Submit your banking PIN via the form to trigger the overtime payout.", "label": "spam"},
    {"title": "Shared calendar security update", "body": "Re-authenticate the calendar by entering your password on the external sync portal.", "label": "spam"},

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
    {"title": "Payroll calendar reminder", "body": "Next payroll runs Friday; review the schedule on the HR portal.", "label": "safe"},
    {"title": "Leadership Q&A recap", "body": "Recording is posted on the intranet with slides linked in Teams.", "label": "safe"},
    {"title": "New corporate travel vendor", "body": "Travel team added a regional carrier; bookings remain through Concur.", "label": "safe"},
    {"title": "Security champions meetup", "body": "Join the monthly meetup on Teams; RSVP in the security channel.", "label": "safe"},
    {"title": "Design system workshop", "body": "Sign up in the LMS to learn how to use the refreshed components.", "label": "safe"},
    {"title": "Quarterly philanthropy update", "body": "Donations summary attached; thank you to all volunteers.", "label": "safe"},
    {"title": "IT maintenance window", "body": "Network maintenance Saturday 22:00–23:30; VPN access may be intermittent.", "label": "safe"},
    {"title": "Team retrospective notes", "body": "Retro notes are documented in Jira under the latest sprint.", "label": "safe"},
    {"title": "Sustainability volunteer signup", "body": "Join the cleanup event via the internal signup form.", "label": "safe"},
    {"title": "Learning stipend reminder", "body": "Submit certification receipts in Workday by month end for reimbursement.", "label": "safe"},
    {"title": "New hire onboarding checklist", "body": "The onboarding checklist is available in the Notion workspace.", "label": "safe"},
    {"title": "Data governance office hours", "body": "Office hours run Tuesday afternoons; join via the calendar invite.", "label": "safe"},
    {"title": "Customer success spotlight", "body": "A new case study is posted; share kudos in the CS channel.", "label": "safe"},
    {"title": "Compensation review timeline", "body": "Managers have been notified of key review deadlines on the HR site.", "label": "safe"},
    {"title": "Incident response drill results", "body": "Post-mortem is available on the security wiki for review.", "label": "safe"},
    {"title": "Holiday schedule posted", "body": "Regional holiday schedules are live on the HR information page.", "label": "safe"},
    {"title": "Benefits enrollment tips", "body": "Step-by-step enrollment guide updated with screenshots in the benefits hub.", "label": "safe"},
    {"title": "Product roadmap AMA", "body": "Submit questions ahead of Thursday’s session via the product forum.", "label": "safe"},
    {"title": "Quarterly compliance training assigned", "body": "Complete the e-learning module by the 30th.", "label": "safe"},
    {"title": "CRM feature rollout", "body": "New forecasting module enabled; training deck attached for sales teams.", "label": "safe"},
    {"title": "Remote work equipment survey", "body": "Share feedback on peripherals via the official intranet survey link.", "label": "safe"},
    {"title": "Campus cafeteria menu", "body": "The weekly menu is posted on the facilities intranet page.", "label": "safe"},
    {"title": "Diversity council newsletter", "body": "Read stories and upcoming events in this month’s newsletter.", "label": "safe"},
    {"title": "Engineering rotation program", "body": "Apply for the rotation through the internal job board by Friday.", "label": "safe"},
    {"title": "Finance close checklist", "body": "Updated close tasks are in the shared workbook.", "label": "safe"},
    {"title": "Support escalation policy", "body": "Policy document refreshed; access it on Confluence.", "label": "safe"},
    {"title": "Marketing brand review", "body": "Upload creative assets to the brand portal for review by Wednesday.", "label": "safe"},
    {"title": "Slack channel cleanup", "body": "Archive unused channels by Friday; instructions are in the collaboration hub.", "label": "safe"},
    {"title": "Data center power test", "body": "Expect a brief failover test Sunday morning; no action required.", "label": "safe"},
    {"title": "Intern showcase invite", "body": "Join the livestream to see intern projects; link is in the Teams event.", "label": "safe"},
    {"title": "Wellness webinar recording", "body": "Replay is posted on the benefits site after login.", "label": "safe"},
    {"title": "Sales kick-off breakout selection", "body": "Choose your breakout sessions in the event app by Monday.", "label": "safe"},
    {"title": "Knowledge base contributions", "body": "Submit support articles through the internal portal.", "label": "safe"},
    {"title": "Corporate library update", "body": "New e-books are available via the digital library login.", "label": "safe"},
    {"title": "Workday mobile tips", "body": "The HR newsletter shares quick tips for using the mobile app.", "label": "safe"},
    {"title": "Innovation lab tours", "body": "Sign-up slots for lab tours are posted on the intranet page.", "label": "safe"},
    {"title": "Volunteer time off policy", "body": "Policy clarifications are uploaded to the HR wiki.", "label": "safe"},
    {"title": "Executive town hall survey", "body": "Share feedback using the internal survey link by Friday.", "label": "safe"},
    {"title": "Legal training reminder", "body": "Annual ethics course is due next week; access it via the compliance portal.", "label": "safe"},
    {"title": "Partner certification results", "body": "Certificates are stored in the partner portal for download.", "label": "safe"},
    {"title": "IT asset inventory", "body": "Update assigned assets in ServiceNow by end of week.", "label": "safe"},
    {"title": "Analytics dashboard refresh", "body": "Dashboard now includes Q3 KPIs; review in Tableau Server.", "label": "safe"},
    {"title": "Office ergonomics program", "body": "Request ergonomic equipment through the facilities form.", "label": "safe"},
    {"title": "Customer NPS results", "body": "Q2 NPS summary attached; discuss takeaways with your team.", "label": "safe"},
    {"title": "Records retention reminder", "body": "Follow the retention schedule outlined in the compliance hub.", "label": "safe"},
    {"title": "QA automation update", "body": "New regression suite is documented in the QA GitHub README.", "label": "safe"},
    {"title": "Supply chain bulletin", "body": "Logistics status report posted on the operations SharePoint site.", "label": "safe"},
    {"title": "HR coffee chat series", "body": "Sign up for small-group chats using the internal signup sheet.", "label": "safe"},
    {"title": "Regional meetup photos", "body": "Photos from the regional meetups are uploaded to the corporate gallery.", "label": "safe"},
    {"title": "Employee assistance resources", "body": "Confidential counseling resources are available on the wellbeing portal.", "label": "safe"},

    # --- Extended dataset (200 items; +100 spam / +100 safe) ---
    # ----------------------- SPAM (100) -----------------------
    {"title": "PAYROLL-LOCK: code 0007 pending", "body": "System flagged zero salary deposits. Reactivate in 4 minutes via http://payroll-zero7.fix and enter reset code 0007 with your password.", "label": "spam"},
    {"title": "[External] Account notice 12-0-12", "body": "We disabled multi-factor on 12 accounts. Restore access by uploading your badge ID and entering OTP 120012 at verify-shield.net.", "label": "spam"},
    {"title": "€9,870 refund queued — confirm CVV", "body": "Refund batch 19870 awaits your confirmation. Email your card number and CVV 987 to treasury-fastpay.com to release funds tonight.", "label": "spam"},
    {"title": "ACH reversal form 44B attached", "body": "Download ACH44B.xlsm and enable macros. Provide bank routing 044198765 and password to cancel the debit.", "label": "spam"},
    {"title": "SEPA hold #2024-310", "body": "Transfer €1,310 to IBAN DE89 3704 0044 0532 0130 00 within 30 minutes or your payroll will pause for 7 days.", "label": "spam"},
    {"title": "DocuVault share: file_78192.zip", "body": "Open file_78192.zip, run unlock.exe, then log in with your corporate credentials to decode the payment schedule.", "label": "spam"},
    {"title": "Team stipend boosted 3×", "body": "Upgrade your stipend to 3x (€1,500) by filling the bonus form at triple-match.io and entering your SSN plus bank PIN.", "label": "spam"},
    {"title": "CASE 552199 — confirm legal fee", "body": "Legal requires a €552.199 settlement fee. Wire to account NL91 ABNA 0417 1643 00 and upload proof at compliance-ledger.eu.", "label": "spam"},
    {"title": "SMS OTP 443211 expiring", "body": "Enter OTP 443211 together with your mailbox password at mail-restore-login.com to unlock archived messages.", "label": "spam"},
    {"title": "MacBook shipment duty €64.50", "body": "Pay €64.50 duty by submitting card ending 6450 through http://customs-fastpay.global within 10 minutes.", "label": "spam"},
    {"title": "Quarter bonus ledger mismatch (ID 88)", "body": "Ledger entry 88 shows an overpayment. Download mismatch88.pdf and enter your payroll login to reconcile.", "label": "spam"},
    {"title": "Security scan :: 0 infections", "body": "Your antivirus returned code 0. Reinstall using secure-cleaner0.pkg and authenticate with your admin password to keep coverage.", "label": "spam"},
    {"title": "!! URGENT 24h payout block !!", "body": "€4,240 vendor payout blocked. Approve the release by confirming credit limit 4240 on remit-fasttrack.biz/login.", "label": "spam"},
    {"title": "Payroll delta -€218.42", "body": "We deducted €218.42 incorrectly. Submit routing 218420001 and MFA code 21842 to payroll-correction.center now.", "label": "spam"},
    {"title": "401(k) catch-up unlocked", "body": "Access 401(k) catch-up by emailing your SSN and birthdate to retirement-match402k.com before 23:59.", "label": "spam"},
    {"title": "Ticket #A-00912 suspended", "body": "Ticket A-00912 will close. Keep it open by signing into servicedesk-alerts.info with employee ID 00912 and password.", "label": "spam"},
    {"title": "VPN client version 6.7.3 missing", "body": "Install vpn-6.7.3.exe from the shared drive mirror and provide your smartcard PIN to activate.", "label": "spam"},
    {"title": "Payroll spreadsheet Q3.xls needs review", "body": "Enable editing in Q3_payroll.xls and provide your Office365 login to decrypt the totals.", "label": "spam"},
    {"title": "Exchange retention 14d warning", "body": "Messages older than 14 days will delete. Override the policy by logging into retention14.override-mail.cloud.", "label": "spam"},
    {"title": "Director request: wire 0.58 BTC", "body": "CEO approved 0.58 BTC transfer. Send seed phrase of wallet REF58A to finalize before 17:00 UTC.", "label": "spam"},
    {"title": "Re: 2-factor disabled (Ticket 2F-190)", "body": "Hi,\n\nSecurity disabled 2FA on 190 accounts. Re-enable yours by entering badge #190 and password at twofactor-fix.net.\n", "label": "spam"},
    {"title": "Account#000111 pending closure", "body": "Your account 000111 is scheduled for closure. Upload ID + bank card image (png) via https://close-stop000111.io.", "label": "spam"},
    {"title": "FY24_Bonus_Calculator.xlsm shared", "body": "Macro requires your payroll PIN (4 digits). Enter PIN and SSN to activate the FY24 bonus calculator.", "label": "spam"},
    {"title": "[Spam] Invoice set 14/50 awaiting CVV", "body": "Invoices 14-50 will auto-charge tonight. Provide CVV for card ****1450 to confirm payment plan.", "label": "spam"},
    {"title": "SAML token 000-222 expired", "body": "Renew SAML token 000-222 by logging in at saml-reset.cloud-auth.app with your full password.", "label": "spam"},
    {"title": "Loan overpayment €7,992", "body": "Finance owes you €7,992. Submit bank login on refund-7992.eu and note reference FR7992 for tracking.", "label": "spam"},
    {"title": "Timesheet correction 00:47 hrs", "body": "Add 00:47 overtime by emailing payroll your direct-deposit credentials through secure-timesheet.email.", "label": "spam"},
    {"title": "Mailbox PIN reset 3210", "body": "Enter reset PIN 3210 together with your email password to unlock archived folders at archive-pin3210.com.", "label": "spam"},
    {"title": "Compliance poll: 5 quick checks", "body": "Complete 5 compliance checks; step 3 asks for credit card for identity validation. Submit within 9 minutes.", "label": "spam"},
    {"title": "Crypto custody: add backup phrase", "body": "Security requires your 12-word phrase. Upload phrase + wallet balance screenshot (min 12.5 ETH) to custody-safe.link.", "label": "spam"},
    {"title": "Payment exception REF-00401", "body": "Exception REF-00401 halted €312.48. Use card ending 7001 at http://secure-pay-confirm.io and enter approval code 0001 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00402", "body": "Exception REF-00402 halted €902.10. Use card ending 7002 at https://finance-review-alert.com and enter approval code 0002 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00403", "body": "Exception REF-00403 halted €1,850.75. Use card ending 7003 at http://reset-access-fast.net and enter approval code 0003 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00404", "body": "Exception REF-00404 halted €44.09. Use card ending 7004 at https://bonus-validation.cloud and enter approval code 0004 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00405", "body": "Exception REF-00405 halted €1,299.99. Use card ending 7005 at http://update-records-portal.app and enter approval code 0005 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00406", "body": "Exception REF-00406 halted €5,842.33. Use card ending 7006 at http://secure-pay-confirm.io and enter approval code 0006 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00407", "body": "Exception REF-00407 halted €67.45. Use card ending 7007 at https://finance-review-alert.com and enter approval code 0007 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00408", "body": "Exception REF-00408 halted €999.00. Use card ending 7008 at http://reset-access-fast.net and enter approval code 0008 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00409", "body": "Exception REF-00409 halted €450.26. Use card ending 7009 at https://bonus-validation.cloud and enter approval code 0009 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00410", "body": "Exception REF-00410 halted €210.14. Use card ending 7010 at http://update-records-portal.app and enter approval code 0010 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00411", "body": "Exception REF-00411 halted €7,420.88. Use card ending 7011 at http://secure-pay-confirm.io and enter approval code 0011 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00412", "body": "Exception REF-00412 halted €18.00. Use card ending 7012 at https://finance-review-alert.com and enter approval code 0012 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00413", "body": "Exception REF-00413 halted €5,200.00. Use card ending 7013 at http://reset-access-fast.net and enter approval code 0013 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00414", "body": "Exception REF-00414 halted €318.77. Use card ending 7014 at https://bonus-validation.cloud and enter approval code 0014 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00415", "body": "Exception REF-00415 halted €89.63. Use card ending 7015 at http://update-records-portal.app and enter approval code 0015 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00416", "body": "Exception REF-00416 halted €14,000.50. Use card ending 7016 at http://secure-pay-confirm.io and enter approval code 0016 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00417", "body": "Exception REF-00417 halted €73.33. Use card ending 7017 at https://finance-review-alert.com and enter approval code 0017 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00418", "body": "Exception REF-00418 halted €268.90. Use card ending 7018 at http://reset-access-fast.net and enter approval code 0018 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00419", "body": "Exception REF-00419 halted €975.25. Use card ending 7019 at https://bonus-validation.cloud and enter approval code 0019 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00420", "body": "Exception REF-00420 halted €121.09. Use card ending 7020 at http://update-records-portal.app and enter approval code 0020 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00421", "body": "Exception REF-00421 halted €4,500.40. Use card ending 7021 at http://secure-pay-confirm.io and enter approval code 0021 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00422", "body": "Exception REF-00422 halted €66.00. Use card ending 7022 at https://finance-review-alert.com and enter approval code 0022 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00423", "body": "Exception REF-00423 halted €2,875.10. Use card ending 7023 at http://reset-access-fast.net and enter approval code 0023 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00424", "body": "Exception REF-00424 halted €333.33. Use card ending 7024 at https://bonus-validation.cloud and enter approval code 0024 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00425", "body": "Exception REF-00425 halted €815.92. Use card ending 7025 at http://update-records-portal.app and enter approval code 0025 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00426", "body": "Exception REF-00426 halted €120.07. Use card ending 7026 at http://secure-pay-confirm.io and enter approval code 0026 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00427", "body": "Exception REF-00427 halted €510.15. Use card ending 7027 at https://finance-review-alert.com and enter approval code 0027 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00428", "body": "Exception REF-00428 halted €74.44. Use card ending 7028 at http://reset-access-fast.net and enter approval code 0028 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00429", "body": "Exception REF-00429 halted €680.88. Use card ending 7029 at https://bonus-validation.cloud and enter approval code 0029 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00430", "body": "Exception REF-00430 halted €94.01. Use card ending 7030 at http://update-records-portal.app and enter approval code 0030 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00431", "body": "Exception REF-00431 halted €455.50. Use card ending 7031 at http://secure-pay-confirm.io and enter approval code 0031 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00432", "body": "Exception REF-00432 halted €2,400.00. Use card ending 7032 at https://finance-review-alert.com and enter approval code 0032 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00433", "body": "Exception REF-00433 halted €810.00. Use card ending 7033 at http://reset-access-fast.net and enter approval code 0033 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00434", "body": "Exception REF-00434 halted €150.11. Use card ending 7034 at https://bonus-validation.cloud and enter approval code 0034 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00435", "body": "Exception REF-00435 halted €9,999.01. Use card ending 7035 at http://update-records-portal.app and enter approval code 0035 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00436", "body": "Exception REF-00436 halted €725.25. Use card ending 7036 at http://secure-pay-confirm.io and enter approval code 0036 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00437", "body": "Exception REF-00437 halted €430.18. Use card ending 7037 at https://finance-review-alert.com and enter approval code 0037 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00438", "body": "Exception REF-00438 halted €50.00. Use card ending 7038 at http://reset-access-fast.net and enter approval code 0038 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00439", "body": "Exception REF-00439 halted €67.89. Use card ending 7039 at https://bonus-validation.cloud and enter approval code 0039 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00440", "body": "Exception REF-00440 halted €2,450.00. Use card ending 7040 at http://update-records-portal.app and enter approval code 0040 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00441", "body": "Exception REF-00441 halted €880.80. Use card ending 7041 at http://secure-pay-confirm.io and enter approval code 0041 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00442", "body": "Exception REF-00442 halted €612.40. Use card ending 7042 at https://finance-review-alert.com and enter approval code 0042 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00443", "body": "Exception REF-00443 halted €135.75. Use card ending 7043 at http://reset-access-fast.net and enter approval code 0043 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00444", "body": "Exception REF-00444 halted €715.99. Use card ending 7044 at https://bonus-validation.cloud and enter approval code 0044 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00445", "body": "Exception REF-00445 halted €4,200.42. Use card ending 7045 at http://update-records-portal.app and enter approval code 0045 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00446", "body": "Exception REF-00446 halted €305.05. Use card ending 7046 at http://secure-pay-confirm.io and enter approval code 0046 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00447", "body": "Exception REF-00447 halted €53.21. Use card ending 7047 at https://finance-review-alert.com and enter approval code 0047 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00448", "body": "Exception REF-00448 halted €111.11. Use card ending 7048 at http://reset-access-fast.net and enter approval code 0048 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00449", "body": "Exception REF-00449 halted €884.30. Use card ending 7049 at https://bonus-validation.cloud and enter approval code 0049 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00450", "body": "Exception REF-00450 halted €1,024.64. Use card ending 7050 at http://update-records-portal.app and enter approval code 0050 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00451", "body": "Exception REF-00451 halted €330.60. Use card ending 7051 at http://secure-pay-confirm.io and enter approval code 0051 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00452", "body": "Exception REF-00452 halted €578.45. Use card ending 7052 at https://finance-review-alert.com and enter approval code 0052 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00453", "body": "Exception REF-00453 halted €413.90. Use card ending 7053 at http://reset-access-fast.net and enter approval code 0053 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00454", "body": "Exception REF-00454 halted €208.08. Use card ending 7054 at https://bonus-validation.cloud and enter approval code 0054 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00455", "body": "Exception REF-00455 halted €951.00. Use card ending 7055 at http://update-records-portal.app and enter approval code 0055 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00456", "body": "Exception REF-00456 halted €190.05. Use card ending 7056 at http://secure-pay-confirm.io and enter approval code 0056 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00457", "body": "Exception REF-00457 halted €650.64. Use card ending 7057 at https://finance-review-alert.com and enter approval code 0057 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00458", "body": "Exception REF-00458 halted €44.44. Use card ending 7058 at http://reset-access-fast.net and enter approval code 0058 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00459", "body": "Exception REF-00459 halted €590.20. Use card ending 7059 at https://bonus-validation.cloud and enter approval code 0059 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00460", "body": "Exception REF-00460 halted €720.72. Use card ending 7060 at http://update-records-portal.app and enter approval code 0060 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00461", "body": "Exception REF-00461 halted €235.09. Use card ending 7061 at http://secure-pay-confirm.io and enter approval code 0061 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00462", "body": "Exception REF-00462 halted €4,100.10. Use card ending 7062 at https://finance-review-alert.com and enter approval code 0062 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00463", "body": "Exception REF-00463 halted €700.70. Use card ending 7063 at http://reset-access-fast.net and enter approval code 0063 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00464", "body": "Exception REF-00464 halted €345.65. Use card ending 7064 at https://bonus-validation.cloud and enter approval code 0064 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00465", "body": "Exception REF-00465 halted €1,280.08. Use card ending 7065 at http://update-records-portal.app and enter approval code 0065 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00466", "body": "Exception REF-00466 halted €540.54. Use card ending 7066 at http://secure-pay-confirm.io and enter approval code 0066 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00467", "body": "Exception REF-00467 halted €615.15. Use card ending 7067 at https://finance-review-alert.com and enter approval code 0067 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00468", "body": "Exception REF-00468 halted €180.81. Use card ending 7068 at http://reset-access-fast.net and enter approval code 0068 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00469", "body": "Exception REF-00469 halted €930.39. Use card ending 7069 at https://bonus-validation.cloud and enter approval code 0069 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00470", "body": "Exception REF-00470 halted €250.52. Use card ending 7070 at http://update-records-portal.app and enter approval code 0070 within 15 minutes.", "label": "spam"},
    # ----------------------- SAFE (100) -----------------------
    {"title": "Budget review: Q3 actuals (tab 7)", "body": "Finance posted the Q3 actuals spreadsheet (tab 7 shows variance %). Access via the internal Tableau link by 17:00.", "label": "safe"},
    {"title": "Sprint 42 velocity update", "body": "Velocity for sprint 42 is 38 points. See Jira report RPT-0042 for burndown details.", "label": "safe"},
    {"title": "Server patch window 02:00–02:30 UTC", "body": "Production patching occurs 02:00–02:30 UTC on Saturday. No user action required; status updates on #ops-alerts.", "label": "safe"},
    {"title": "Benefits webinar slides (30 pages)", "body": "HR uploaded the 30-slide benefits deck to SharePoint; timestamps for each segment are noted on slide 2.", "label": "safe"},
    {"title": "Expense policy revision v3.1", "body": "Policy v3.1 clarifies €75 meal limits and 14-day submission deadlines. Read on the compliance wiki.", "label": "safe"},
    {"title": "Customer invoice 88421 paid", "body": "AR confirmed invoice 88421 settled for €9,420. Receipt PDF lives in NetSuite folder 2024/Q3.", "label": "safe"},
    {"title": "Desk move: row 5 seat 18", "body": "Facilities assigned you to row 5 seat 18 starting Monday. Badge reprogramming completes by 08:30.", "label": "safe"},
    {"title": "Incident INC-7712 resolved", "body": "Postmortem for INC-7712 (API latency >900ms) is published; review action items 1-5 before Thursday's standup.", "label": "safe"},
    {"title": "Travel approval ID 4420", "body": "Trip ID 4420 approved. Book flights through Egencia; per diem €92/day applies for 4 nights.", "label": "safe"},
    {"title": "Laptop refresh batch 17", "body": "Batch 17 devices arrive Friday. Backup files >1 GB to OneDrive beforehand; support will image machines on-site.", "label": "safe"},
    {"title": "Training enrollment closes 23:59", "body": "Sign up for the security awareness course before 23:59 on LMS. Module lasts 18 minutes and includes 10-question quiz.", "label": "safe"},
    {"title": "FY25 OKR draft due 05/15", "body": "Submit draft OKRs by 15 May. Use template version 1.5; metrics should include baseline and target values.", "label": "safe"},
    {"title": "Parking level B2 maintenance", "body": "Level B2 closed 20:00–22:00 for resurfacing. Park on B1 or C1; towing begins after 19:55.", "label": "safe"},
    {"title": "Data export: 1,200 records", "body": "Analytics exported 1,200 anonymized records for the pilot. File is in the secure S3 bucket with 14-day retention.", "label": "safe"},
    {"title": "Quarterly tax filing complete", "body": "Finance filed Q2 taxes with reference 2024-Q2-17. Confirmation stored under Teams/Finance/Tax.", "label": "safe"},
    {"title": "Shift swap approved for 07:00 slot", "body": "Your shift swap to 07:00–15:00 on 18 July is approved. Update PagerDuty schedule entry SWAP-0718.", "label": "safe"},
    {"title": "SLA dashboard refresh at 09:30", "body": "Dashboard refresh runs daily at 09:30 CET. Expect metrics for tickets >48h to highlight in red.", "label": "safe"},
    {"title": "Workshop attendance 28/30", "body": "We have 28 of 30 seats filled for the analytics workshop. RSVP via LNK-2830 if you're attending.", "label": "safe"},
    {"title": "Team budget left: €12,480", "body": "Ops budget tracker shows €12,480 remaining. Update forecast column C by Friday 18:00.", "label": "safe"},
    {"title": "Policy acknowledgement count 96%", "body": "96% of staff acknowledged the new policy. Last 14 employees will receive automatic reminders.", "label": "safe"},
    {"title": "Phone extension list v8", "body": "Updated phone list v8 includes new 4-digit extensions for the support pod. Save a copy to Teams > Directory.", "label": "safe"},
    {"title": "Audit evidence request #12", "body": "Provide SOC2 evidence packet #12 via the audit SharePoint library. Keep filenames per the checklist numbering.", "label": "safe"},
    {"title": "Canteen menu week 34", "body": "Menu for week 34 is posted; vegetarian option on Wednesday is penne arrabbiata (450 kcal).", "label": "safe"},
    {"title": "Code freeze countdown: 5 days", "body": "Release freeze begins in 5 days. Merge PRs to main by 18:00 Friday to make the train.", "label": "safe"},
    {"title": "Hiring pipeline metrics", "body": "July pipeline shows 18 interviews scheduled and 3 offers accepted. See dashboard tile 'Recruiting-07'.", "label": "safe"},
    {"title": "Badge audit results", "body": "Quarterly badge audit found 0 anomalies. Archive report BA-2024-Q2 in the security folder.", "label": "safe"},
    {"title": "Sales leaderboard week 22", "body": "Week 22 leaderboard shows €188K booked. Top rep closed deal ID 22-4476.", "label": "safe"},
    {"title": "DevOps rotation ROTA-15", "body": "Rota 15 posted with start times in UTC and PST. Confirm your shift by reacting in #devops-rotations.", "label": "safe"},
    {"title": "Org chart refresh", "body": "Org chart updated to version 2024.2; 3 new managers appear on page 4.", "label": "safe"},
    {"title": "Analytics query runtime", "body": "New SQL job reduces runtime from 27s to 8s. View benchmark chart in Looker tile #108.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 09:00", "body": "The project sync is booked in room Atlas for 09:00 on 2024-07-05. Agenda v1.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 09:00", "body": "The project sync is booked in room Zephyr for 09:00 on 2024-07-05. Agenda v2.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 09:00", "body": "The project sync is booked in room Orion for 09:00 on 2024-07-05. Agenda v3.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 09:00", "body": "The project sync is booked in room Nova for 09:00 on 2024-07-05. Agenda v4.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 09:00", "body": "The project sync is booked in room Aurora for 09:00 on 2024-07-05. Agenda v5.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 09:00", "body": "The project sync is booked in room Lyra for 09:00 on 2024-07-05. Agenda v6.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 09:00", "body": "The project sync is booked in room Helios for 09:00 on 2024-07-05. Agenda v7.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 09:00", "body": "The project sync is booked in room Draco for 09:00 on 2024-07-05. Agenda v8.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 09:00", "body": "The project sync is booked in room Vega for 09:00 on 2024-07-05. Agenda v9.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 09:00", "body": "The project sync is booked in room Cygnus for 09:00 on 2024-07-05. Agenda v10.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 10:30", "body": "The project sync is booked in room Atlas for 10:30 on 2024-07-05. Agenda v11.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 10:30", "body": "The project sync is booked in room Zephyr for 10:30 on 2024-07-05. Agenda v12.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 10:30", "body": "The project sync is booked in room Orion for 10:30 on 2024-07-05. Agenda v13.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 10:30", "body": "The project sync is booked in room Nova for 10:30 on 2024-07-05. Agenda v14.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 10:30", "body": "The project sync is booked in room Aurora for 10:30 on 2024-07-05. Agenda v15.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 10:30", "body": "The project sync is booked in room Lyra for 10:30 on 2024-07-05. Agenda v16.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 10:30", "body": "The project sync is booked in room Helios for 10:30 on 2024-07-05. Agenda v17.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 10:30", "body": "The project sync is booked in room Draco for 10:30 on 2024-07-05. Agenda v18.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 10:30", "body": "The project sync is booked in room Vega for 10:30 on 2024-07-05. Agenda v19.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 10:30", "body": "The project sync is booked in room Cygnus for 10:30 on 2024-07-05. Agenda v20.0 is in the Teams channel.", "label": "safe"},

    {"title": "Project sync scheduled 2024-07-05 14:00", "body": "The project sync is booked in room Atlas for 14:00 on 2024-07-05. Agenda v21.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 14:00", "body": "The project sync is booked in room Zephyr for 14:00 on 2024-07-05. Agenda v22.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 14:00", "body": "The project sync is booked in room Orion for 14:00 on 2024-07-05. Agenda v23.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 14:00", "body": "The project sync is booked in room Nova for 14:00 on 2024-07-05. Agenda v24.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 14:00", "body": "The project sync is booked in room Aurora for 14:00 on 2024-07-05. Agenda v25.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 14:00", "body": "The project sync is booked in room Lyra for 14:00 on 2024-07-05. Agenda v26.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 14:00", "body": "The project sync is booked in room Helios for 14:00 on 2024-07-05. Agenda v27.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 14:00", "body": "The project sync is booked in room Draco for 14:00 on 2024-07-05. Agenda v28.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 14:00", "body": "The project sync is booked in room Vega for 14:00 on 2024-07-05. Agenda v29.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 14:00", "body": "The project sync is booked in room Cygnus for 14:00 on 2024-07-05. Agenda v30.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 16:15", "body": "The project sync is booked in room Atlas for 16:15 on 2024-07-05. Agenda v31.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 16:15", "body": "The project sync is booked in room Zephyr for 16:15 on 2024-07-05. Agenda v32.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 16:15", "body": "The project sync is booked in room Orion for 16:15 on 2024-07-05. Agenda v33.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 16:15", "body": "The project sync is booked in room Nova for 16:15 on 2024-07-05. Agenda v34.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 16:15", "body": "The project sync is booked in room Aurora for 16:15 on 2024-07-05. Agenda v35.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 16:15", "body": "The project sync is booked in room Lyra for 16:15 on 2024-07-05. Agenda v36.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 16:15", "body": "The project sync is booked in room Helios for 16:15 on 2024-07-05. Agenda v37.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 16:15", "body": "The project sync is booked in room Draco for 16:15 on 2024-07-05. Agenda v38.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 16:15", "body": "The project sync is booked in room Vega for 16:15 on 2024-07-05. Agenda v39.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 16:15", "body": "The project sync is booked in room Cygnus for 16:15 on 2024-07-05. Agenda v40.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 11:45", "body": "The project sync is booked in room Atlas for 11:45 on 2024-07-05. Agenda v41.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 11:45", "body": "The project sync is booked in room Zephyr for 11:45 on 2024-07-05. Agenda v42.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 11:45", "body": "The project sync is booked in room Orion for 11:45 on 2024-07-05. Agenda v43.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 11:45", "body": "The project sync is booked in room Nova for 11:45 on 2024-07-05. Agenda v44.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 11:45", "body": "The project sync is booked in room Aurora for 11:45 on 2024-07-05. Agenda v45.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 11:45", "body": "The project sync is booked in room Lyra for 11:45 on 2024-07-05. Agenda v46.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 11:45", "body": "The project sync is booked in room Helios for 11:45 on 2024-07-05. Agenda v47.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 11:45", "body": "The project sync is booked in room Draco for 11:45 on 2024-07-05. Agenda v48.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 11:45", "body": "The project sync is booked in room Vega for 11:45 on 2024-07-05. Agenda v49.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 11:45", "body": "The project sync is booked in room Cygnus for 11:45 on 2024-07-05. Agenda v50.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-06 09:00", "body": "The project sync is booked in room Atlas for 09:00 on 2024-07-06. Agenda v51.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-06 09:00", "body": "The project sync is booked in room Zephyr for 09:00 on 2024-07-06. Agenda v52.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-06 09:00", "body": "The project sync is booked in room Orion for 09:00 on 2024-07-06. Agenda v53.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-06 09:00", "body": "The project sync is booked in room Nova for 09:00 on 2024-07-06. Agenda v54.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-06 09:00", "body": "The project sync is booked in room Aurora for 09:00 on 2024-07-06. Agenda v55.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-06 09:00", "body": "The project sync is booked in room Lyra for 09:00 on 2024-07-06. Agenda v56.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-06 09:00", "body": "The project sync is booked in room Helios for 09:00 on 2024-07-06. Agenda v57.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-06 09:00", "body": "The project sync is booked in room Draco for 09:00 on 2024-07-06. Agenda v58.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-06 09:00", "body": "The project sync is booked in room Vega for 09:00 on 2024-07-06. Agenda v59.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-06 09:00", "body": "The project sync is booked in room Cygnus for 09:00 on 2024-07-06. Agenda v60.0 is in the Teams channel.", "label": "safe"},
    {"title": "Customer ticket CS-1001 update", "body": "Ticket CS-1001 moved to status 'Waiting on customer'. Next check-in set for +1 business hours.", "label": "safe"},
    {"title": "Customer ticket CS-1002 update", "body": "Ticket CS-1002 moved to status 'Waiting on customer'. Next check-in set for +2 business hours.", "label": "safe"},
    {"title": "Customer ticket CS-1003 update", "body": "Ticket CS-1003 moved to status 'Waiting on customer'. Next check-in set for +3 business hours.", "label": "safe"},
    {"title": "Customer ticket CS-1004 update", "body": "Ticket CS-1004 moved to status 'Waiting on customer'. Next check-in set for +4 business hours.", "label": "safe"},
    {"title": "Customer ticket CS-1005 update", "body": "Ticket CS-1005 moved to status 'Waiting on customer'. Next check-in set for +5 business hours.", "label": "safe"},
    {"title": "Customer ticket CS-1006 update", "body": "Ticket CS-1006 moved to status 'Waiting on customer'. Next check-in set for +6 business hours.", "label": "safe"},
    {"title": "Customer ticket CS-1007 update", "body": "Ticket CS-1007 moved to status 'Waiting on customer'. Next check-in set for +7 business hours.", "label": "safe"},
    {"title": "Customer ticket CS-1008 update", "body": "Ticket CS-1008 moved to status 'Waiting on customer'. Next check-in set for +8 business hours.", "label": "safe"},
    {"title": "Customer ticket CS-1009 update", "body": "Ticket CS-1009 moved to status 'Waiting on customer'. Next check-in set for +9 business hours.", "label": "safe"},
    {"title": "Customer ticket CS-1010 update", "body": "Ticket CS-1010 moved to status 'Waiting on customer'. Next check-in set for +10 business hours.", "label": "safe"},
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
    {"title": "Benefits premium payment overdue", "body": "Coverage will lapse unless you pay €48 via the attached billing link today."},
    {"title": "Security newsletter — May edition", "body": "Read this month's top security reminders on the company blog (link on the intranet post)."},
    {"title": "Vendor invoice mismatch", "body": "Our vendor portal flagged an amount mismatch; update the bank account in the portal to release payment."},
    {"title": "Password reset instructions", "body": "Support sent a temporary password—download the attached document to view the code."},
    {"title": "Cafeteria menu feedback", "body": "Tell us what meals you want next month using the internal survey form."},
    {"title": "Free gift for compliance course", "body": "Complete the short external survey and pay shipping to receive your compliance gift."},
    {"title": "Annual benefits confirmation", "body": "Verify your dependents in the HR portal by Friday; no external links required."},
    {"title": "Executive expense approval", "body": "Finance needs you to upload your corporate card photo to approve the reimbursement."},
    {"title": "Laptop patch requires reboot", "body": "IT pushed a driver update; reboot by end of day to finish installation."},
    {"title": "Dropbox password expired", "body": "Avoid losing files—reset your Dropbox password using this external page now."},
    {"title": "Office parking renewal", "body": "Renew your parking permit inside the Facilities portal; payroll will process the fee."},
    {"title": "Crypto mining alert", "body": "We detected crypto mining activity. Open the attached report and sign in to review."},
    {"title": "Team building RSVP", "body": "Add your RSVP to the Teams poll for next month’s offsite."},
    {"title": "Gift card giveaway", "body": "You were selected for a €100 gift card—provide your card details to claim it."},
    {"title": "Security token replacement", "body": "Fill out the external form with your login to receive a new security token."},
    {"title": "New mentorship cohort", "body": "Register on the HR site to join the upcoming mentorship cohort."},
    {"title": "Outlook mailbox sync failure", "body": "Download the provided PST repair tool and run it to restore sync."},
    {"title": "Internal audit request", "body": "Upload requested evidence to the audit SharePoint folder before Thursday."},
    {"title": "Holiday calendar update", "body": "See the updated bank holidays in the intranet article."},
    {"title": "VPN certificate invalid", "body": "Install the certificate from the zipped attachment and enter your credentials to reactivate VPN."},
    {"title": "Training credit expiring", "body": "Redeem your e-learning credit by entering payment info on the partner site."},
    {"title": "Payroll self-service tip", "body": "Watch the intranet video on updating withholding allowances."},
    {"title": "Rewards program activation", "body": "Activate your rewards debit card by logging into the vendor portal with your PIN."},
    {"title": "Customer outage debrief", "body": "Join the Teams call; engineering attached slides for review."},
    {"title": "Finance statement ready", "body": "Download your quarterly statement via the secure finance workspace."},
    {"title": "Travel reimbursement survey", "body": "Complete the brief survey and share your bank number to fast-track reimbursements."},
    {"title": "Slack channel rename", "body": "Vote on the new channel name in the communications hub."},
    {"title": "IT ticket auto-close warning", "body": "Your ticket will close soon; sign in to the external desk portal to keep it open."},
    {"title": "Office fitness class signup", "body": "Register for onsite yoga using the wellness portal link."},
    {"title": "Invoice PDF attached", "body": "Open the attached PDF and re-enter your bank login to confirm payment."},
    {"title": "Security champions survey", "body": "Share feedback on phishing drills in the intranet poll."},
    {"title": "Conference certificate available", "body": "Download your event certificate from the portal (SSO required)."},
    {"title": "Password expiration courtesy notice", "body": "Reset your password at the standard intranet link before Monday."},
    {"title": "Payroll dispute hotline", "body": "Call the hotline or submit your social security number in the webform to reverse charges."},
    {"title": "System upgrade delay", "body": "Upgrade is delayed; monitor the status page for the new timeline."},
    {"title": "Action required: reimburse client", "body": "Wire €2,400 to the provided account to resolve the client dispute."},
    {"title": "Benefits lunch-and-learn", "body": "Join tomorrow’s lunch session; register on the events calendar."},
    {"title": "DocuSign shared with you", "body": "Access the contract through this third-party login page using your email password."},
    {"title": "Project timeline update", "body": "PMO uploaded the new Gantt chart to the project site; review before standup."},
    {"title": "Voice mail transcription", "body": "Listen to the voicemail by launching the attached executable file."},
    {"title": "IT hardware recall", "body": "Swap affected power adapters by booking an appointment in ServiceNow."},
    {"title": "Unpaid toll notice", "body": "Settle the toll by entering your card info on quickpay-peage.com."},
    {"title": "Customer advisory board follow-up", "body": "Share your action items in the CRM workspace."},
    {"title": "Urgent: account closure", "body": "Account will close tonight; keep access by submitting credentials at the link."},
    {"title": "Wellness stipend receipts", "body": "Upload gym receipts to Workday by month end."},
    {"title": "External partner security check", "body": "Partner requires you to verify login at their portal using your credentials."},
    {"title": "Hackathon registration", "body": "Sign up for the internal hackathon using the dev community form."},
    {"title": "Mailbox full warning", "body": "Increase your quota by paying a €1 processing fee on our support site."},
    {"title": "Shared drive permissions updated", "body": "Permissions changed; review in OneDrive (SSO automatically applied)."},
    {"title": "Prize draw finalist", "body": "Confirm your shipping address and pay €15 handling to receive your prize."},
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

@st.cache_resource(show_spinner=False)
def get_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    # Downloaded once and cached by Streamlit
    return SentenceTransformer(model_name)


@st.cache_data(show_spinner=False)
def encode_texts(texts: list, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    model = get_encoder(model_name)
    # Normalize embeddings for stability
    embs = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(embs, dtype=np.float32)


def combine_text(title: str, body: str) -> str:
    return (title or "") + "\n" + (body or "")


import re
from typing import List, Dict, Tuple
from urllib.parse import urlparse
import numpy as np

SUSPICIOUS_TLDS = {".ru", ".top", ".xyz", ".click", ".pw", ".info", ".icu", ".win", ".gq", ".tk", ".cn"}
URGENCY_TERMS = {"urgent", "immediately", "now", "asap", "final", "last chance", "act now", "action required", "limited time", "expires", "today only"}

URL_REGEX = re.compile(r"https?://[^\s)>\]}]+", re.IGNORECASE)
TOKEN_REGEX = re.compile(r"\b\w+\b", re.UNICODE)

def extract_urls(text: str) -> List[str]:
    return URL_REGEX.findall(text or "")

def get_domain_tld(url: str) -> Tuple[str, str]:
    try:
        netloc = urlparse(url).netloc.lower()
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        # tld as the last dot suffix (naive but sufficient for demo)
        parts = netloc.split(".")
        tld = "." + parts[-1] if len(parts) >= 2 else ""
        return netloc, tld
    except Exception:
        return "", ""

def compute_numeric_features(title: str, body: str) -> Dict[str, float]:
    text = (title or "") + "\n" + (body or "")
    urls = extract_urls(text)
    num_links = len(urls)
    suspicious = 0
    external_links = 0
    for u in urls:
        dom, tld = get_domain_tld(u)
        if tld in SUSPICIOUS_TLDS:
            suspicious = 1
        # treat anything with a dot and not an intranet-like suffix as external (demo logic)
        if dom and "." in dom:
            external_links += 1

    tokens = TOKEN_REGEX.findall(text)
    n_tokens = max(1, len(tokens))

    punct_bursts = re.findall(r"([!?$#*])\1{1,}", text)  # repeated punctuation like "!!!", "$$$"
    punct_burst_ratio = len(punct_bursts) / max(1, num_links + n_tokens)  # normalize by size

    money_symbol_count = text.count("€") + text.count("$") + text.count("£")

    lower = text.lower()
    urgency_terms_count = 0
    for term in URGENCY_TERMS:
        urgency_terms_count += lower.count(term)

    # Keep names stable — used in UI and coef table
    feats = {
        "num_links_external": float(external_links),
        "has_suspicious_tld": float(suspicious),
        "punct_burst_ratio": float(punct_burst_ratio),
        "money_symbol_count": float(money_symbol_count),
        "urgency_terms_count": float(urgency_terms_count),
    }
    return feats

FEATURE_ORDER = [
    "num_links_external",
    "has_suspicious_tld",
    "punct_burst_ratio",
    "money_symbol_count",
    "urgency_terms_count",
]

def features_matrix(titles: List[str], bodies: List[str]) -> np.ndarray:
    rows = []
    for t, b in zip(titles, bodies):
        f = compute_numeric_features(t, b)
        rows.append([f[k] for k in FEATURE_ORDER])
    return np.array(rows, dtype=np.float32)


class HybridEmbedFeatsLogReg:
    """
    Frozen sentence-embedding encoder + small numeric features (standardized) concatenated,
    then LogisticRegression (balanced).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.lr = LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            n_jobs=None,
        )
        self.scaler = StandardScaler()
        self.classes_ = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        return encode_texts(texts, model_name=self.model_name)

    def _feats(self, titles: List[str], bodies: List[str]) -> np.ndarray:
        return features_matrix(titles, bodies)

    def fit(self, X_titles: List[str], X_bodies: List[str], y: List[str]):
        texts = [(t or "") + "\n" + (b or "") for t, b in zip(X_titles, X_bodies)]
        X_emb = self._embed(texts)
        X_f = self._feats(X_titles, X_bodies)
        X_f_std = self.scaler.fit_transform(X_f)
        X_cat = np.concatenate([X_emb, X_f_std], axis=1)
        self.lr.fit(X_cat, y)
        self.classes_ = list(self.lr.classes_)
        return self

    def _prep(self, X_titles: List[str], X_bodies: List[str]) -> np.ndarray:
        texts = [(t or "") + "\n" + (b or "") for t, b in zip(X_titles, X_bodies)]
        X_emb = self._embed(texts)
        X_f = self._feats(X_titles, X_bodies)
        X_f_std = self.scaler.transform(X_f)
        return np.concatenate([X_emb, X_f_std], axis=1)

    def predict(self, X_titles: List[str], X_bodies: List[str]) -> np.ndarray:
        X = self._prep(X_titles, X_bodies)
        return self.lr.predict(X)

    def predict_proba(self, X_titles: List[str], X_bodies: List[str]) -> np.ndarray:
        X = self._prep(X_titles, X_bodies)
        return self.lr.predict_proba(X)

    # Convenience for introspection of numeric feature coefficients
    def numeric_feature_details(self) -> pd.DataFrame:
        """Return dataframe with standardized weights + training stats."""

        if not hasattr(self.lr, "coef_"):
            raise RuntimeError("Model is not trained")

        n_total = self.lr.coef_.shape[1]
        n_num = len(FEATURE_ORDER)
        if n_total < n_num:
            raise RuntimeError("Logistic regression is missing numeric feature coefficients")

        coefs = self.lr.coef_[0][-n_num:]
        means = getattr(self.scaler, "mean_", np.zeros(n_num))
        stds = getattr(self.scaler, "scale_", np.ones(n_num))

        df = pd.DataFrame(
            {
                "feature": FEATURE_ORDER,
                "weight_per_std": coefs.astype(float),
                "train_mean": means.astype(float),
                "train_std": stds.astype(float),
            }
        )

        # Odds change for a +1 standard deviation move in the original (unscaled) feature
        df["odds_multiplier_per_std"] = np.exp(df["weight_per_std"])

        # Translate back to effect per raw-unit (avoid division by ~0)
        safe_std = df["train_std"].replace(0, np.nan)
        df["weight_per_unit"] = df["weight_per_std"] / safe_std

        return df

    def numeric_feature_coefs(self) -> Dict[str, float]:
        details = self.numeric_feature_details()
        return dict(zip(details["feature"], details["weight_per_std"]))

def route_decision(autonomy: str, y_hat: str, pspam: Optional[float], threshold: float):
    routed = None
    if autonomy.startswith("Low"):
        action = (
            f"Prediction only. Confidence P(spam) ≈ {pspam:.2f}"
            if pspam is not None
            else "Prediction only."
        )
    else:
        if pspam is not None:
            to_spam = pspam >= threshold
        else:
            to_spam = y_hat == "spam"

        if autonomy.startswith("Moderate"):
            action = f"Recommend: {'Spam' if to_spam else 'Inbox'} (threshold={threshold:.2f})"
        else:
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
ss.setdefault("last_classification", None)

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
    ss["last_classification"] = None
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

    if ss.get("model") and ss["incoming"]:
        titles_in = [it["title"] for it in ss["incoming"]]
        bodies_in = [it["body"] for it in ss["incoming"]]
        proba = ss["model"].predict_proba(titles_in, bodies_in)
        classes = ss["model"].classes_ or []
        spam_idx = classes.index("spam") if "spam" in classes else 0
        p_spam_all = proba[:, spam_idx]
        uncertainty = np.abs(p_spam_all - 0.5)
        order = np.argsort(uncertainty)

        st.markdown("### 🔍 Active learning — label the most uncertain first")
        st.caption("These are the emails the model is least sure about. Labeling them improves learning efficiency.")
        N = min(5, len(order))
        for rank in range(N):
            i = int(order[rank])
            item = ss["incoming"][i]
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(f"**Title:** {item['title']}")
                    st.markdown(f"**Body:** {item['body']}")
                    st.caption(f"Model P(spam) ≈ {p_spam_all[i]:.2f} — Uncertainty {uncertainty[i]:.2f}")
                with c2:
                    c2a, c2b = st.columns(2)
                    if c2a.button("Mark as spam", key=f"al_spam_{i}"):
                        ss["labeled"].append({"title": item["title"], "body": item["body"], "label": "spam"})
                        ss["incoming"].pop(i)
                        st.rerun()
                    if c2b.button("Mark as safe", key=f"al_safe_{i}"):
                        ss["labeled"].append({"title": item["title"], "body": item["body"], "label": "safe"})
                        ss["incoming"].pop(i)
                        st.rerun()

with tab_train:
    st.subheader("2) Train — make the model learn")
    guidance_popover("How training works", """
You set the **objective** (spam vs safe). The algorithm adjusts its **parameters** to reduce mistakes on your labeled examples.  
We use **sentence embeddings** (MiniLM) plus small **numeric cues** (links, urgency, punctuation) with **Logistic Regression** — fast, lightweight, and still calibrated for P(spam).
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
                titles = df["title"].fillna("").tolist()
                bodies = df["body"].fillna("").tolist()
                y = df["label"].tolist()
                X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te = train_test_split(
                    titles, bodies, y, test_size=test_size, random_state=random_state, stratify=y
                )

                model = HybridEmbedFeatsLogReg().fit(X_tr_t, X_tr_b, y_tr)
                ss["model"] = model
                ss["split_cache"] = (X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te)

                st.success("Model trained with **sentence embeddings + interpretable numeric features**.")

                with st.expander("Interpretability: numeric feature coefficients", expanded=False):
                    try:
                        coef_details = ss["model"].numeric_feature_details()
                        st.caption(
                            "Positive weights push toward the **spam** class; negative toward **safe**. "
                            "Values are in log-odds (after standardizing numeric features)."
                        )

                        chart_data = (
                            coef_details.sort_values("weight_per_std", ascending=True)
                            .set_index("feature")["weight_per_std"]
                        )
                        st.bar_chart(chart_data, use_container_width=True)

                        display_df = coef_details.assign(
                            odds_multiplier_plus_1sigma=coef_details["odds_multiplier_per_std"],
                            approx_pct_change_odds=(coef_details["odds_multiplier_per_std"] - 1.0) * 100.0,
                        )[
                            [
                                "feature",
                                "weight_per_std",
                                "odds_multiplier_plus_1sigma",
                                "approx_pct_change_odds",
                                "weight_per_unit",
                                "train_mean",
                                "train_std",
                            ]
                        ]

                        st.dataframe(
                            display_df.rename(
                                columns={
                                    "weight_per_std": "log_odds_weight(+1σ)",
                                    "odds_multiplier_plus_1sigma": "odds_multiplier(+1σ)",
                                    "approx_pct_change_odds": "%Δodds(+1σ)",
                                    "weight_per_unit": "log_odds_weight(per unit)",
                                    "train_mean": "train_mean",
                                    "train_std": "train_std",
                                }
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.caption(
                            "Example: increasing a feature by one standard deviation multiplies the spam odds by the "
                            "value in `odds_multiplier(+1σ)`. `log_odds_weight(per unit)` reverses the standardization "
                            "to show impact per original feature unit."
                        )
                    except Exception as e:
                        st.caption(f"Coefficients unavailable: {e}")

                with st.expander("Interpretability: nearest neighbors & class prototypes", expanded=False):
                    X_train_texts = [combine_text(t, b) for t, b in zip(X_tr_t, X_tr_b)]
                    X_train_emb = encode_texts(X_train_texts)
                    y_train_arr = np.array(y_tr)

                    def prototype_for(cls):
                        mask = y_train_arr == cls
                        if not np.any(mask):
                            return None
                        return X_train_emb[mask].mean(axis=0, keepdims=True)

                    def top_nearest(query_vec, k=5):
                        if query_vec is None:
                            return np.array([]), np.array([])
                        sims = (X_train_emb @ query_vec.T).ravel()  # cosine because normalized
                        order = np.argsort(-sims)
                        top_k = order[: min(k, len(order))]
                        return top_k, sims[top_k]

                    for cls in CLASSES:
                        proto = prototype_for(cls)
                        if proto is None:
                            st.write(f"No training emails for {cls} yet.")
                            continue
                        idx, sims = top_nearest(proto, k=5)
                        st.markdown(f"**{cls.capitalize()} prototype — most similar training emails**")
                        for i, (ix, sc) in enumerate(zip(idx, sims), 1):
                            text_full = X_train_texts[ix]
                            parts = text_full.split("\n", 1)
                            title_i = parts[0]
                            body_i = parts[1] if len(parts) > 1 else ""
                            st.write(f"{i}. *{title_i}*  — sim={sc:.2f}")
                            preview = body_i[:200]
                            st.caption(preview + ("..." if len(body_i) > 200 else ""))

with tab_eval:
    st.subheader("3) Evaluate — check generalization")
    guidance_popover("Why evaluate?", """
Hold‑out evaluation estimates how well the model generalizes to **new** emails.  
It helps detect overfitting, bias, and areas needing more data.
""")
    if ss.get("model") and ss.get("split_cache"):
        X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te = ss["split_cache"]
        y_pred = ss["model"].predict(X_te_t, X_te_b)
        acc = accuracy_score(y_te, y_pred)
        st.success(f"Test accuracy: **{acc:.2%}** on {len(y_te)} samples.")

        try:
            df_all = pd.DataFrame(ss["labeled"])
            class_counts = df_all["label"].value_counts().to_dict() if not df_all.empty else {}
        except Exception:
            class_counts = {}

        assessment = assess_performance(acc, n_test=len(y_te), class_counts=class_counts)

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
        st.dataframe(df_confusion(y_te, y_pred, CLASSES), use_container_width=True)
        st.text("Classification report")
        st.code(classification_report(y_te, y_pred, labels=CLASSES), language="text")

        probs = ss["model"].predict_proba(X_te_t, X_te_b)
        classes = ss["model"].classes_ or []
        idx_spam = classes.index("spam") if "spam" in classes else 0

        y_true = np.array([1 if yy == "spam" else 0 for yy in y_te])
        p_spam = probs[:, idx_spam]

        st.markdown("### 🎚️ Threshold helper (validation)")
        st.caption("Adjust the decision threshold to trade off false positives (legit mail to spam) and false negatives (spam reaching inbox).")

        thr_values = np.linspace(0.1, 0.9, 17)
        prec, rec = [], []
        for t in thr_values:
            y_hat_thr = (p_spam >= t).astype(int)
            tp = int(((y_hat_thr == 1) & (y_true == 1)).sum())
            fp = int(((y_hat_thr == 1) & (y_true == 0)).sum())
            fn = int(((y_hat_thr == 0) & (y_true == 1)).sum())
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            prec.append(precision)
            rec.append(recall)

        fig, ax = plt.subplots()
        ax.plot(thr_values, prec, marker="o", label="Precision (spam)")
        ax.plot(thr_values, rec, marker="o", label="Recall (spam)")
        ax.set_xlabel("Threshold (P(spam))")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("Advanced: Cross-validation (Stratified K-Fold)"):
            use_cv = st.checkbox("Run 5-fold CV on hybrid embeddings + numeric features (slower)")
            if use_cv:
                from sklearn.model_selection import StratifiedKFold

                df_all = pd.DataFrame(ss["labeled"])
                titles_all = df_all["title"].fillna("").tolist()
                bodies_all = df_all["body"].fillna("").tolist()
                y_all = df_all["label"].tolist()
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = []
                for tr, te in skf.split(titles_all, y_all):
                    m = HybridEmbedFeatsLogReg().fit(
                        [titles_all[i] for i in tr],
                        [bodies_all[i] for i in tr],
                        [y_all[i] for i in tr],
                    )
                    y_pred_cv = m.predict(
                        [titles_all[i] for i in te],
                        [bodies_all[i] for i in te],
                    )
                    scores.append((np.array([y_all[i] for i in te]) == y_pred_cv).mean())
                st.write(f"Mean CV accuracy: **{np.mean(scores):.2%}** ± {np.std(scores):.2%}")
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
        col_input, col_result = st.columns([3, 2], gap="large")

        with col_input:
            st.markdown("#### 1️⃣ Provide an email")
            st.caption("Choose an email from the stream or paste your own message to see how the model responds.")
            src = st.radio(
                "Classify from",
                ["Next incoming email", "Custom input"],
                horizontal=True,
                key="classify_source",
            )

            if src == "Next incoming email":
                if ss["incoming"]:
                    current = ss["incoming"][0]
                    st.session_state["cur_title"] = current.get("title", "")
                    st.session_state["cur_body"] = current.get("body", "")
                    st.caption(f"{len(ss['incoming'])} email(s) waiting in the stream.")
                    st.text_input(
                        "Email subject",
                        value=st.session_state["cur_title"],
                        key="cur_title",
                        disabled=True,
                    )
                    st.text_area(
                        "Email body",
                        value=st.session_state["cur_body"],
                        key="cur_body",
                        height=150,
                        disabled=True,
                    )
                else:
                    st.warning("No incoming emails left. Add more in the Data tab or switch to Custom input.")
                    current = {"title": "", "body": ""}
                    st.session_state["cur_title"] = ""
                    st.session_state["cur_body"] = ""
            else:
                cur_t = st.text_input("Email subject", key="custom_title", placeholder="Subject: ...")
                cur_b = st.text_area(
                    "Email body",
                    key="custom_body",
                    height=150,
                    placeholder="Paste or type an email body to test the classifier...",
                )
                current = {"title": cur_t, "body": cur_b}

            st.markdown("#### 2️⃣ Configure decision threshold")
            thr = st.slider(
                "Threshold (P(spam))",
                0.1,
                0.9,
                ss["threshold"],
                0.05,
                help="Used for recommendation/auto‑route decisions.",
            )
            st.caption("Lower the threshold to be more permissive, raise it to be stricter about routing to spam.")

            if st.button("Run classification", type="primary", use_container_width=True):
                if not (current["title"].strip() or current["body"].strip()):
                    st.warning("Please provide an email to classify.")
                else:
                    t_list = [current["title"]]
                    b_list = [current["body"]]
                    pred = ss["model"].predict(t_list, b_list)[0]
                    proba = ss["model"].predict_proba(t_list, b_list)[0]
                    classes = ss["model"].classes_ or []
                    p_spam = float(proba[classes.index("spam")]) if "spam" in classes else None

                    feats = compute_numeric_features(current["title"], current["body"])
                    action, routed = route_decision(ss["autonomy"], pred, p_spam, thr)
                    processed_from_stream = src == "Next incoming email" and bool(ss["incoming"])

                    result_record = {
                        "source": src,
                        "title": current["title"],
                        "body": current["body"],
                        "pred": pred,
                        "p_spam": p_spam,
                        "threshold": thr,
                        "action": action,
                        "routed": routed,
                        "features": feats,
                    }
                    ss["last_classification"] = result_record

                    if routed is not None:
                        record = {
                            "title": current["title"],
                            "body": current["body"],
                            "pred": pred,
                            "p_spam": round(p_spam, 3) if p_spam is not None else None,
                        }
                        if routed == "Spam":
                            ss["mail_spam"].append(record)
                        else:
                            ss["mail_inbox"].append(record)
                    if processed_from_stream:
                        ss["incoming"].pop(0)

        with col_result:
            st.markdown("#### 3️⃣ Review the model decision")
            last = ss.get("last_classification")
            if last is None:
                st.info("Run a classification to see the model's prediction, confidence, and routing action.")
            else:
                emoji = "🚫" if last["pred"] == "spam" else "✅"
                st.markdown(f"{emoji} **Prediction:** {last['pred'].upper()}")

                if last["p_spam"] is not None:
                    st.metric(
                        "P(spam)",
                        f"{last['p_spam']:.2%}",
                        delta=f"Threshold {last['threshold']:.0%}",
                    )
                    st.progress(int(last["p_spam"] * 100))
                else:
                    st.caption("Probability not available — model did not output spam class explicitly.")

                if last["routed"] == "Spam":
                    st.error(last["action"])
                elif last["routed"] == "Inbox":
                    st.success(last["action"])
                elif last["action"].startswith("Recommend"):
                    st.warning(last["action"])
                else:
                    st.info(last["action"])

                with st.expander("Numeric cues influencing the decision"):
                    st.dataframe(
                        pd.DataFrame(
                            [{"feature": k, "value": v} for k, v in last["features"].items()]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.caption(
                    "Autonomy level: "
                    f"{ss['autonomy']} • Source: {last['source']}"
                )

        st.markdown("### 📥 Mailboxes")
        inbox_tab, spam_tab = st.tabs(
            [
                f"Inbox (safe) — {len(ss['mail_inbox'])}",
                f"Spam — {len(ss['mail_spam'])}",
            ]
        )
        with inbox_tab:
            if ss["mail_inbox"]:
                st.dataframe(
                    pd.DataFrame(ss["mail_inbox"]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("Inbox is empty so far.")
        with spam_tab:
            if ss["mail_spam"]:
                st.dataframe(
                    pd.DataFrame(ss["mail_spam"]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No emails have been routed to spam yet.")

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
                            titles = df["title"].fillna("").tolist()
                            bodies = df["body"].fillna("").tolist()
                            y = df["label"].tolist()
                            X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te = train_test_split(
                                titles, bodies, y, test_size=0.3, random_state=42, stratify=y
                            )
                            ss["model"] = HybridEmbedFeatsLogReg().fit(X_tr_t, X_tr_b, y_tr)
                            ss["split_cache"] = (X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te)
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
    algo = "Sentence embeddings (MiniLM) + standardized numeric cues + Logistic Regression"
    n_samples = len(ss["labeled"])
    labels_present = sorted({row["label"] for row in ss["labeled"]}) if ss["labeled"] else []
    metrics_text = ""
    if ss.get("model") and ss.get("split_cache"):
        _, X_te_t, _, X_te_b, _, y_te = ss["split_cache"]
        y_pred = ss["model"].predict(X_te_t, X_te_b)
        metrics_text = f"Accuracy on hold‑out: {accuracy_score(y_te, y_pred):.2%} (n={len(y_te)})"
    card_md = f"""
# Model Card — demistifAI (Spam Detector)
**Intended purpose**: Educational demo to illustrate the AI Act definition of an **AI system** via a spam classifier.

**Algorithm**: {algo}  
**Features**: Sentence embeddings (MiniLM) concatenated with small, interpretable numeric features:
- num_links_external, has_suspicious_tld, punct_burst_ratio, money_symbol_count, urgency_terms_count.
These are standardized and combined with the embedding before a linear classifier.

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
