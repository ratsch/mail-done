# Review System Test Suite

## Overview

Comprehensive test suite for the Lab Application Review System Phase 2 implementation.

## Test Files

- `test_review_auth.py` - Unit tests for authentication logic
- `test_review_endpoints.py` - API endpoint tests
- `test_review_auth_flow.py` - OAuth authentication flow tests
- `test_review_security.py` - Security tests (SQL injection, XSS, rate limiting)
- `test_review_edge_cases.py` - Edge cases and validation tests
- `test_review_filters.py` - Advanced filtering and sorting tests
- `test_review_integration.py` - Integration tests (triggers, workflows)
- `test_deployed_api.py` - Tests against deployed API (requires deployment)

## Running Tests

### Local Tests (No External Dependencies)

```bash
# Run all local tests
poetry run pytest backend/tests/review/ -v -k "not test_deployed"

# Run specific test file
poetry run pytest backend/tests/review/test_review_endpoints.py -v

# Run with coverage
poetry run pytest backend/tests/review/ --cov=backend/api --cov-report=html -k "not test_deployed"
```

### With Encryption

Tests use environment variables for configuration. Set encryption key if testing encrypted fields:

```bash
export DB_ENCRYPTION_KEY=your_encryption_key_here
poetry run pytest backend/tests/review/ -v
```

**Note**: The encryption key should be set as an environment variable, never hardcoded in test files.

### Deployed API Tests

Requires deployed API and credentials:

```bash
export API_URL="https://your-deployed-api-url"
export TEST_TOKEN="<jwt_token>"
poetry run pytest backend/tests/review/test_deployed_api.py -v
```

Or use the test script:

```bash
./scripts/run_deployed_tests.sh
```

## Environment Variables

### Required for Local Tests
- `TEST_DATABASE_URL` - Test database URL (defaults to SQLite in-memory)
- `JWT_SECRET` - JWT secret key (defaults to "test_secret" for tests)

### Optional
- `DB_ENCRYPTION_KEY` - Encryption key for testing encrypted fields
- `GOOGLE_CLIENT_ID` - Google OAuth client ID (for real OAuth tests)
- `GOOGLE_CLIENT_SECRET` - Google OAuth client secret (for real OAuth tests)

### For Deployed API Tests
- `API_URL` - Deployed API URL
- `TEST_TOKEN` - Valid JWT token for testing

## Test Coverage

### Current Coverage
- ✅ **56 tests passing** (local tests)
- ⏭️ **7 tests skipped** (deployed API tests)
- **Coverage**: ~90%+ for core functionality

### Test Categories

1. **Authentication** (6 tests)
   - JWT creation/validation
   - Token expiration/blacklisting
   - OAuth state management
   - Account lockout

2. **API Endpoints** (40+ tests)
   - Application listing/filtering/sorting
   - Application details
   - Review submission/management
   - Admin endpoints
   - Export endpoints
   - Notifications
   - Statistics

3. **Security** (5+ tests)
   - SQL injection prevention
   - XSS prevention
   - Input validation
   - Rate limiting
   - PII protection

4. **Edge Cases** (26+ tests)
   - Invalid inputs
   - Boundary conditions
   - Error handling
   - Validation

5. **Filters** (10+ tests)
   - Advanced filtering
   - Sorting options
   - Filter combinations

6. **Integration** (8+ tests)
   - Database triggers
   - Complex workflows
   - Concurrency

## Test Fixtures

Key fixtures available in `conftest.py`:

- `test_db` - Test database session
- `client` - FastAPI test client
- `admin_user`, `reviewer_user`, `regular_user` - User fixtures
- `admin_token`, `reviewer_token`, `regular_token` - JWT tokens
- `multiple_applications` - Sample application emails
- `sample_application` - Single application fixture

## Known Limitations

1. **SQLite vs PostgreSQL**: Some PostgreSQL-specific features (JSON operators, triggers) may not work in SQLite tests
2. **Google Drive**: GDrive integration is mocked - real integration tests require credentials
3. **OAuth**: Real OAuth tests require Google credentials (currently mocked)
4. **Triggers**: Database triggers may not fire in SQLite (tests document expected behavior)

## Adding New Tests

When adding new tests:

1. Use existing fixtures where possible
2. Set up test data in fixtures or test functions
3. Clean up after tests (fixtures handle this automatically)
4. Use descriptive test names
5. Test both success and failure cases
6. Document any limitations or assumptions

## Troubleshooting

### Tests failing with encryption errors
- Set `DB_ENCRYPTION_KEY` environment variable
- Ensure key matches the one used in production

### Tests failing with database errors
- Check `TEST_DATABASE_URL` is set correctly
- Ensure database is accessible
- SQLite in-memory is used by default (no setup needed)

### OAuth tests failing
- OAuth tests are mocked by default
- For real OAuth tests, set `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`
- Real OAuth tests may be skipped if credentials not available

### Import errors
- Ensure you're running from project root
- Check that `backend` is in Python path
- Run `poetry install` to ensure dependencies are installed
