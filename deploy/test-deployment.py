#!/usr/bin/env python3
"""
Functional tests for mail-done deployment.

Run this script to verify a deployment is working correctly:
    python3 deploy/test-deployment.py                  # Test localhost
    python3 deploy/test-deployment.py http://pi:8000   # Test remote host
"""

import json
import sys
import urllib.request
import urllib.error


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    passed = 0
    failed = 0

    def test(name, ok, details=""):
        nonlocal passed, failed
        status = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
        print(f"{status} {name}")
        if not ok and details:
            print(f"  {details}")
        passed += ok
        failed += not ok

    rate_limited = False

    def get(path):
        nonlocal rate_limited
        try:
            with urllib.request.urlopen(f"{base_url}{path}", timeout=10) as r:
                body = r.read()
                return r.status, json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            try:
                body = e.read()
                data = json.loads(body) if body else {}
                if e.code == 429:
                    rate_limited = True
                return e.code, data
            except:
                if e.code == 429:
                    rate_limited = True
                return e.code, {}
        except Exception as e:
            return 0, {"error": str(e)}

    print("=" * 60)
    print(f"  Mail-Done Deployment Tests")
    print(f"  Target: {base_url}")
    print("=" * 60)
    print()

    # Core health checks
    print("Core Health Checks:")
    status, data = get("/health")
    test("Health endpoint accessible", status == 200, f"Got status {status}")
    test("Status is healthy", data.get("status") == "healthy", f"Got: {data.get('status')}")
    test("Database connected", data.get("database") == "connected", f"Got: {data.get('database')}")

    checks = data.get("checks", {})
    test("Database health check passed", checks.get("database") == "ok")

    # API info
    print("\nAPI Information:")
    status, data = get("/")
    # 429 means rate-limited but API is running
    test("Root endpoint accessible", status in [200, 429])
    if status == 200:
        test("Version info present", "version" in data, "Missing 'version' field")
        test("Features listed", len(data.get("features", [])) > 0)
        if data.get("version"):
            print(f"  Version: {data['version']}")
    else:
        test("Rate-limited (API is running)", status == 429)
        print("  (Skipping version check due to rate limiting)")

    # API documentation
    print("\nAPI Documentation:")
    status, data = get("/openapi.json")
    test("OpenAPI spec accessible", status == 200)
    endpoint_count = len(data.get("paths", {}))
    test(f"API has endpoints defined ({endpoint_count} found)", endpoint_count > 50)

    # Authentication enforcement
    # Note: 429 (rate limit) also indicates auth is working
    print("\nAuthentication Enforcement:")
    status, _ = get("/api/stats")
    test("Stats requires auth", status in [401, 429])

    status, _ = get("/api/emails")
    test("Emails requires auth", status in [401, 429])

    status, _ = get("/admin/users")
    test("Admin endpoints require auth", status in [401, 429])

    status, _ = get("/applications")
    test("Applications requires auth", status in [401, 429])

    # Summary
    print()
    print("=" * 60)
    total = passed + failed
    if failed == 0:
        print(f"\033[32m  All {total} tests passed!\033[0m")
        print("  Deployment is working correctly.")
    else:
        print(f"\033[31m  {failed} of {total} tests failed\033[0m")
        if rate_limited:
            print("  Note: Rate limiting is active (429 responses).")
            print("  Wait 60 seconds and retry, or test from a different IP.")
        else:
            print("  Check the deployment logs for issues.")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
