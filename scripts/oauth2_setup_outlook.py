#!/usr/bin/env python3
"""
Microsoft 365 OAuth2 Setup Script

Interactive OAuth2 login to get refresh token for IMAP access.
Run this once to get the initial refresh token, then store it in .env.

Usage:
    python3 scripts/oauth2_setup_outlook.py
    python3 scripts/oauth2_setup_outlook.py --tenant-id <tenant> --client-id <client>

See docs/OUTLOOK_OAUTH2.md for full setup instructions.
"""
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="Get OAuth2 refresh token for Microsoft 365 IMAP access",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode (prompts for IDs)
    python3 scripts/oauth2_setup_outlook.py

    # With arguments
    python3 scripts/oauth2_setup_outlook.py --tenant-id organizations --client-id abc123...

    # For personal Microsoft account
    python3 scripts/oauth2_setup_outlook.py --tenant-id consumers --client-id abc123...

See docs/OUTLOOK_OAUTH2.md for full setup instructions.
        """
    )
    parser.add_argument(
        "--tenant-id",
        help="Azure AD tenant ID (e.g., 'organizations', 'consumers', or specific GUID)"
    )
    parser.add_argument(
        "--client-id",
        help="Application (client) ID from Azure AD app registration"
    )
    parser.add_argument(
        "--client-secret",
        help="Optional client secret (not needed for public client flow)"
    )
    parser.add_argument(
        "--account-name",
        default="OFFICE365",
        help="Account name for .env variable naming (default: OFFICE365)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  Microsoft 365 OAuth2 Setup for mail-done")
    print("=" * 60)
    print()
    print("This script will open a browser for you to sign in to your")
    print("Microsoft account. After signing in, you'll receive a refresh")
    print("token to store in your .env file.")
    print()
    print("Prerequisites:")
    print("  1. Azure AD app registered (see docs/OUTLOOK_OAUTH2.md)")
    print("  2. IMAP.AccessAsUser.All permission added")
    print("  3. Public client flow enabled")
    print()

    # Get tenant ID
    tenant_id = args.tenant_id
    if not tenant_id:
        print("Tenant ID options:")
        print("  - 'organizations' : Any organizational account (recommended)")
        print("  - 'consumers'     : Personal Microsoft accounts only")
        print("  - 'common'        : Both organizational and personal")
        print("  - <GUID>          : Specific organization only")
        print()
        tenant_id = input("Enter tenant ID [organizations]: ").strip() or "organizations"

    # Get client ID
    client_id = args.client_id
    if not client_id:
        print()
        print("Find your Application (client) ID in:")
        print("  Azure Portal → App registrations → Your app → Overview")
        print()
        client_id = input("Enter Application (client) ID: ").strip()
        if not client_id:
            print("Error: Client ID is required")
            sys.exit(1)

    client_secret = args.client_secret
    account_name = args.account_name.upper().replace("-", "_").replace(" ", "_")

    print()
    print("-" * 60)
    print(f"Tenant ID:    {tenant_id}")
    print(f"Client ID:    {client_id}")
    print(f"Account name: {account_name}")
    print("-" * 60)
    print()

    # Check for MSAL
    try:
        import msal
    except ImportError:
        print("Error: MSAL library not installed")
        print()
        print("Install with:")
        print("  poetry add msal")
        print("  # or")
        print("  pip install msal")
        sys.exit(1)

    # Import after path setup
    try:
        from backend.core.auth.oauth2_provider import get_initial_tokens_interactive
    except ImportError as e:
        print(f"Error importing oauth2_provider: {e}")
        print()
        print("Make sure you're running from the mail-done project directory:")
        print("  cd /path/to/mail-done")
        print("  python3 scripts/oauth2_setup_outlook.py")
        sys.exit(1)

    print("Opening browser for Microsoft sign-in...")
    print("(If browser doesn't open, check for popup blocker)")
    print()

    try:
        result = get_initial_tokens_interactive(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret
        )

        refresh_token = result.get("refresh_token")
        account_email = result.get("id_token_claims", {}).get("preferred_username", "")

        if not refresh_token:
            print()
            print("Warning: No refresh token received!")
            print("This might happen if:")
            print("  - The app doesn't have offline_access scope")
            print("  - The account type doesn't support refresh tokens")
            print()
            print("Full response:")
            for key in result:
                if key != "access_token":
                    print(f"  {key}: {result[key]}")
            sys.exit(1)

        print()
        print("=" * 60)
        print("  SUCCESS!")
        print("=" * 60)
        print()
        print(f"Account: {account_email}")
        print()
        print("Add these to your .env file:")
        print()
        print(f"IMAP_USERNAME_{account_name}={account_email}")
        print(f"OAUTH2_REFRESH_TOKEN_{account_name}={refresh_token}")
        print()
        print("-" * 60)
        print()
        print("Add to your accounts.yaml:")
        print()
        print(f"""  {account_name.lower()}:
    display_name: "Microsoft 365"
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
      tenant_id: {tenant_id}
      client_id: {client_id}
    folders:
      inbox: INBOX
      sent: Sent Items
      archive: Archive
      trash: Deleted Items""")
        print()
        print("=" * 60)
        print("  Setup complete! Test with:")
        print(f"  python3 process_inbox.py --account {account_name.lower()} --dry-run --limit 1")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"Error: {e}")
        print()
        print("Common issues:")
        print("  - App not registered correctly")
        print("  - Missing IMAP.AccessAsUser.All permission")
        print("  - Public client flow not enabled")
        print()
        print("See docs/OUTLOOK_OAUTH2.md for troubleshooting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
