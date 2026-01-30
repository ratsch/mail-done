# Microsoft 365 / Outlook OAuth2 Setup

This guide covers setting up mail-done with Microsoft 365, Outlook.com, and organizational accounts (universities, enterprises) using OAuth2 authentication.

## Related Documentation

- [Deployment Guide](DEPLOYMENT.md) - Full deployment instructions
- [Gmail Setup](GMAIL_SETUP.md) - For Google accounts

---

## Why OAuth2?

Microsoft has deprecated basic authentication (username/password) for IMAP:
- **Outlook.com**: Basic auth disabled since 2023
- **Microsoft 365**: Basic auth disabled for most tenants
- **OAuth2 is required** for all Microsoft email access

---

## Overview

Setting up OAuth2 for Microsoft 365 involves:

1. **Register an Azure AD application** (one-time setup)
2. **Configure API permissions** for IMAP access
3. **Run interactive login** to get a refresh token
4. **Store the refresh token** in your `.env`

The refresh token lasts 90 days but auto-renews on each use, so it effectively never expires if used regularly.

---

## Step 1: Register an Azure AD Application

### For Personal Microsoft Account (Outlook.com, Hotmail)

1. Go to [Azure Portal - App Registrations](https://portal.azure.com/#blade/Microsoft_AAD_RegisteredApps/ApplicationsListBlade)
2. Sign in with your Microsoft account
3. Click **New registration**
4. Configure:
   - **Name**: `mail-done` (or any name)
   - **Supported account types**: "Personal Microsoft accounts only"
   - **Redirect URI**: Select "Public client/native" → `http://localhost`
5. Click **Register**
6. Copy the **Application (client) ID** - you'll need this

### For Organizational Account (Microsoft 365, University, Enterprise)

This includes ETH, universities, and company Microsoft 365 accounts.

1. Go to [Azure Portal - App Registrations](https://portal.azure.com/#blade/Microsoft_AAD_RegisteredApps/ApplicationsListBlade)
2. Sign in with your organizational account
3. Click **New registration**
4. Configure:
   - **Name**: `mail-done-imap`
   - **Supported account types**: Choose based on your needs:
     - "Accounts in this organizational directory only" (single tenant - most secure)
     - "Accounts in any organizational directory" (multi-tenant)
   - **Redirect URI**: Select "Public client/native" → `http://localhost`
5. Click **Register**
6. Note down:
   - **Application (client) ID**
   - **Directory (tenant) ID**

### ETH Zurich Specific Notes

For ETH accounts (`@ethz.ch`, `@student.ethz.ch`):

- **Tenant ID**: Use `organizations` or ETH's specific tenant ID
- You may need IT approval for app registrations
- Alternative: Check if ETH IT provides a pre-approved OAuth app for IMAP
- Contact: https://ethz.ch/staffnet/en/it-services.html

---

## Step 2: Configure API Permissions

1. In your app registration, go to **API permissions**
2. Click **Add a permission**
3. Select **APIs my organization uses**
4. Search for and select **Office 365 Exchange Online**
5. Select **Delegated permissions**
6. Add these permissions:
   - `IMAP.AccessAsUser.All` - Read/write mailbox via IMAP
   - `SMTP.Send` - Send emails (optional, for drafts)
7. Click **Add permissions**

Your permissions should look like:

| API | Permission | Type | Status |
|-----|------------|------|--------|
| Office 365 Exchange Online | IMAP.AccessAsUser.All | Delegated | Granted |
| Office 365 Exchange Online | SMTP.Send | Delegated | Granted |

**Note:** Admin consent is NOT required for these delegated permissions.

---

## Step 3: Enable Public Client Flow

1. In your app registration, go to **Authentication**
2. Scroll to **Advanced settings**
3. Set **Allow public client flows** to **Yes**
4. Click **Save**

This enables the device code flow and interactive login without a client secret.

---

## Step 4: Get Initial Refresh Token

Create a script to perform the initial interactive login:

```python
#!/usr/bin/env python3
"""
scripts/oauth2_setup_outlook.py

Interactive OAuth2 login to get refresh token for Microsoft 365 IMAP.
Run this once to get the initial refresh token.
"""
import sys
sys.path.insert(0, '.')

from backend.core.auth.oauth2_provider import get_initial_tokens_interactive

# Replace with your values from Step 1
TENANT_ID = "organizations"  # or your specific tenant ID
CLIENT_ID = "your-application-client-id-here"

# For public client (no secret needed for delegated auth)
CLIENT_SECRET = None

if __name__ == "__main__":
    print("=" * 60)
    print("Microsoft 365 OAuth2 Setup for mail-done")
    print("=" * 60)
    print()
    print(f"Tenant ID: {TENANT_ID}")
    print(f"Client ID: {CLIENT_ID}")
    print()
    print("A browser window will open for you to sign in.")
    print("After signing in, the refresh token will be displayed.")
    print()

    try:
        result = get_initial_tokens_interactive(
            tenant_id=TENANT_ID,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET
        )

        print()
        print("=" * 60)
        print("SUCCESS! Add to your .env file:")
        print("=" * 60)
        print()
        print(f"OAUTH2_REFRESH_TOKEN_OFFICE365={result.get('refresh_token', 'N/A')}")
        print()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
```

Run the script:

```bash
cd /path/to/mail-done

# Install MSAL if not already installed
poetry add msal

# Run the setup script
python3 scripts/oauth2_setup_outlook.py
```

A browser window opens → Sign in with your Microsoft account → Copy the refresh token.

---

## Step 5: Configure mail-done

### Environment Variables (.env)

```bash
# Microsoft 365 / Outlook OAuth2
IMAP_USERNAME_OFFICE365=your.email@organization.com
OAUTH2_REFRESH_TOKEN_OFFICE365=0.AXEA...very-long-token...
```

### Account Configuration (accounts.yaml)

```yaml
accounts:
  office365:
    display_name: "Microsoft 365"
    auth_type: oauth2  # This is critical!
    imap:
      host: outlook.office365.com
      port: 993
      use_ssl: true
    smtp:
      host: smtp.office365.com
      port: 587
      use_tls: true
    oauth2:
      tenant_id: organizations  # or your specific tenant ID
      client_id: your-application-client-id
      # Credentials loaded from environment:
      #   IMAP_USERNAME_OFFICE365
      #   OAUTH2_REFRESH_TOKEN_OFFICE365
    folders:
      inbox: INBOX
      sent: Sent Items
      archive: Archive
      trash: Deleted Items
      drafts: Drafts
      junk: Junk Email
```

---

## Step 6: Test Connection

```bash
# Test the connection
python3 process_inbox.py --account office365 --dry-run --limit 1
```

Expected output:
```
INFO - Acquiring new OAuth2 access token for your.email@organization.com
INFO - Successfully acquired OAuth2 token (expires in 3600s)
INFO - Successfully logged in via OAuth2 as your.email@organization.com
```

---

## Tenant ID Reference

| Account Type | Tenant ID |
|--------------|-----------|
| Personal (Outlook.com, Hotmail) | `consumers` |
| Any organizational account | `organizations` |
| Specific organization only | Your tenant's GUID |
| Both personal and organizational | `common` |

To find your organization's tenant ID:
1. Sign in to [Azure Portal](https://portal.azure.com)
2. Go to **Azure Active Directory**
3. **Overview** → **Tenant ID**

---

## Token Refresh Behavior

- **Access tokens** expire after 1 hour
- **Refresh tokens** expire after 90 days of inactivity
- mail-done automatically refreshes tokens before expiry
- If you run mail-done at least once every 90 days, the token never expires
- If token expires, re-run the setup script

---

## Troubleshooting

### "AADSTS50011: Reply URL does not match"

- Ensure redirect URI is set to `http://localhost` (not https)
- Check Authentication settings in app registration

### "AADSTS65001: User or admin has not consented"

For organizational accounts:
- Try signing out and signing in again
- Ask your IT admin to grant consent for the app
- Check if your organization requires admin approval for apps

### "AADSTS7000218: Request body must contain client_secret"

- Enable "Allow public client flows" in app Authentication settings
- Or add a client secret and set `OAUTH2_CLIENT_SECRET_OFFICE365` in .env

### "Login failed" after token refresh

- Token may have expired (90 days of inactivity)
- Re-run `scripts/oauth2_setup_outlook.py` to get a new token

### Connection works locally but not in deployment

- Ensure the same credentials are in the deployment `.env`
- Check CONFIG_DIR points to correct config overlay
- Verify the deployed container can reach `login.microsoftonline.com`

---

## Security Best Practices

1. **Use specific tenant ID** instead of `organizations` when possible
2. **Don't commit tokens** - Keep in `.env` only
3. **Use a dedicated app registration** - Easy to revoke if compromised
4. **Monitor sign-ins** in Azure AD → Sign-in logs
5. **Set token lifetime policies** if your org allows

---

## ETH Zurich Quick Setup

For ETH accounts specifically:

```yaml
# accounts.yaml
accounts:
  eth:
    display_name: "ETH Mail"
    auth_type: oauth2
    imap:
      host: outlook.office365.com
      port: 993
      use_ssl: true
    smtp:
      host: smtp.office365.com
      port: 587
      use_tls: true
    oauth2:
      tenant_id: organizations  # Works for ETH
      client_id: <your-app-client-id>
    folders:
      inbox: INBOX
      sent: Sent Items
      archive: Archive
      trash: Deleted Items
```

```bash
# .env
IMAP_USERNAME_ETH=username@ethz.ch
OAUTH2_REFRESH_TOKEN_ETH=<from setup script>
```

---

## Quick Checklist

- [ ] Azure AD app registered
- [ ] API permissions added (IMAP.AccessAsUser.All)
- [ ] Public client flow enabled
- [ ] Setup script run, refresh token obtained
- [ ] `.env` has `IMAP_USERNAME_*` and `OAUTH2_REFRESH_TOKEN_*`
- [ ] `accounts.yaml` has `auth_type: oauth2` and `oauth2:` section
- [ ] Test connection works
