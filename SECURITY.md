# Security policy

OralAID handles clinical images, so we take security reports seriously.

## Reporting a vulnerability

Report privately through GitHub: open the **Security** tab of this repository, then choose
**Report a vulnerability**. Please include steps to reproduce and the impact you expect.
Don't open a public issue and don't test against other people's data.

We aim to acknowledge reports within 3 working days.

## Supported versions

Only the latest commit on the `prod` branch (the deployed version) receives security fixes.

## What's already in place

- bcrypt password hashing, password policy, account lockout and rate limiting
- JWT sessions with server-side revocation, and API keys stored only as hashes
- HMAC-signed, expiring URLs for every clinical image, plus an audit log of sensitive actions
- strict Content-Security-Policy, HSTS and anti-framing headers
- upload checks that sniff file contents, not file names
- model weights checksum-verified and loaded with `weights_only`
- automated scans on every change: gitleaks, Bandit, pip-audit, npm audit, CodeQL, Trivy and OWASP ZAP
