# Gmail & Google Workspace IMAP Setup

This guide covers setting up mail-done with Gmail (personal) and Google Workspace accounts.

## Related Documentation

- [Deployment Guide](DEPLOYMENT.md) - Full deployment instructions
- [Outlook/Microsoft 365 Setup](OUTLOOK_OAUTH2.md) - For Microsoft accounts

---

## Quick Reference

| Account Type | Auth Method | 2FA Required | Admin Approval |
|--------------|-------------|--------------|----------------|
| Gmail (Personal) | App Password | Yes | No |
| Google Workspace | App Password | Usually | Maybe |
| Google Workspace | OAuth2 | No | Yes |

---

## Option 1: App Passwords (Recommended)

App Passwords are the simplest method and work for both personal Gmail and most Google Workspace accounts.

### Prerequisites

- **2FA must be enabled** on your Google account
- For Workspace: Admin must allow "Less secure apps" or App Passwords

### Step 1: Enable 2-Factor Authentication

1. Go to [Google Account Security](https://myaccount.google.com/security)
2. Under "How you sign in to Google", click **2-Step Verification**
3. Follow the setup wizard (phone number, authenticator app, etc.)

### Step 2: Create an App Password

1. Go to [App Passwords](https://myaccount.google.com/apppasswords)
   - Direct link: https://myaccount.google.com/apppasswords
   - Or: Google Account → Security → 2-Step Verification → App passwords

2. If you don't see "App passwords":
   - 2FA might not be enabled
   - Your Workspace admin may have disabled it
   - Try the OAuth2 method instead

3. Create a new App Password:
   - Select app: **Mail**
   - Select device: **Other (Custom name)**
   - Enter: `mail-done` or similar
   - Click **Generate**

4. **Copy the 16-character password** (shown with spaces, but use without spaces)
   - Example: `abcd efgh ijkl mnop` → use `abcdefghijklmnop`
   - This is shown only once!

### Step 3: Configure mail-done

Add to your `.env`:

```bash
# Gmail credentials
IMAP_USERNAME_GMAIL=your.email@gmail.com
IMAP_PASSWORD_GMAIL=abcdefghijklmnop  # App Password (no spaces)
```

Add to `config/accounts.yaml` (or your private overlay):

```yaml
accounts:
  gmail:
    display_name: "Personal Gmail"
    imap:
      host: imap.gmail.com
      port: 993
      use_ssl: true
    smtp:
      host: smtp.gmail.com
      port: 587
      use_tls: true
    folders:
      inbox: INBOX
      sent: "[Gmail]/Sent Mail"
      archive: "[Gmail]/All Mail"
      trash: "[Gmail]/Trash"
      drafts: "[Gmail]/Drafts"
      spam: "[Gmail]/Spam"
```

### Step 4: Test Connection

```bash
# Quick test
python3 process_inbox.py --account gmail --dry-run --limit 1

# Or test IMAP directly
python3 -c "
import imaplib
m = imaplib.IMAP4_SSL('imap.gmail.com')
m.login('your.email@gmail.com', 'abcdefghijklmnop')
print('Success!')
m.logout()
"
```

---

## Google Workspace (GSuite) Specifics

### Admin-Controlled Settings

Your Workspace admin controls several settings that affect IMAP access:

| Setting | Location | Impact |
|---------|----------|--------|
| IMAP access | Admin Console → Apps → Google Workspace → Gmail → End User Access | Must be ON |
| Less secure apps | Admin Console → Security → Less secure apps | Affects older auth |
| App Passwords | Tied to 2FA enforcement | Usually available if 2FA is on |

### If App Passwords Don't Work

Contact your Workspace admin to verify:
1. IMAP is enabled for your account
2. App Passwords are allowed
3. No conditional access policies blocking IMAP

### Alternative: OAuth2 for Workspace

If your organization requires OAuth2:

1. **Admin must create a GCP project** with Gmail API enabled
2. **Admin must configure OAuth consent screen** (internal)
3. **Create OAuth2 credentials** (Desktop app type)
4. Use Google's OAuth2 flow to get refresh token

This is significantly more complex than App Passwords. See Google's documentation:
- [Gmail API Quickstart](https://developers.google.com/gmail/api/quickstart/python)
- [OAuth 2.0 for Desktop Apps](https://developers.google.com/identity/protocols/oauth2/native-app)

---

## Gmail Folder Names

Gmail uses special folder names. Use these in your `accounts.yaml`:

| Folder | Gmail IMAP Name |
|--------|-----------------|
| Inbox | `INBOX` |
| Sent | `[Gmail]/Sent Mail` |
| Drafts | `[Gmail]/Drafts` |
| Trash | `[Gmail]/Trash` |
| Spam | `[Gmail]/Spam` |
| All Mail | `[Gmail]/All Mail` |
| Starred | `[Gmail]/Starred` |
| Important | `[Gmail]/Important` |

**Note:** Labels appear as folders. A label "Work/Projects" appears as `Work/Projects`.

---

## Troubleshooting

### "Application-specific password required"

- You're using your regular password instead of an App Password
- Generate an App Password as described above

### "Please log in via your web browser"

- Google detected unusual activity
- Log into Gmail in a browser, complete any security checks
- Try again

### "IMAP is disabled"

For personal Gmail:
1. Go to Gmail Settings → See all settings
2. Forwarding and POP/IMAP tab
3. Enable IMAP

For Workspace:
- Contact your admin

### "Username and password not accepted"

1. Verify you're using the App Password, not your regular password
2. Ensure no spaces in the App Password
3. Check username is the full email address

### "Too many simultaneous connections"

Gmail limits to 15 concurrent IMAP connections:
- Close other email clients temporarily
- Check for stuck connections

---

## Security Notes

1. **App Passwords bypass 2FA** - Treat them like your main password
2. **Revoke unused App Passwords** at https://myaccount.google.com/apppasswords
3. **Don't commit App Passwords** - Keep them in `.env` only
4. **Use a dedicated App Password** for mail-done (easy to revoke if needed)

---

## Quick Checklist

- [ ] 2FA enabled on Google account
- [ ] App Password generated and saved
- [ ] IMAP enabled (Settings → Forwarding and POP/IMAP)
- [ ] `.env` has `IMAP_USERNAME_GMAIL` and `IMAP_PASSWORD_GMAIL`
- [ ] `accounts.yaml` has Gmail configuration with correct folder names
- [ ] Test connection works
