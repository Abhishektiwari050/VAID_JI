# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| < Latest| :x:                |

## Reporting a Vulnerability

We take the security of VAID JI seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Please DO NOT:
- Open a public GitHub issue for security vulnerabilities
- Discuss the vulnerability publicly before it has been addressed

### Please DO:
1. **Report privately**: Create a private security advisory via GitHub's security tab
2. **Provide details**: Include as much information as possible:
   - Type of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

3. **Allow time**: Give us reasonable time to respond and fix the issue before public disclosure

### What to expect:
- **Acknowledgment**: Within 48 hours
- **Assessment**: Within 1 week
- **Fix timeline**: Depends on severity
- **Public disclosure**: Coordinated with you

## Security Best Practices for Users

### API Keys
- ✅ Store API keys in `.env` file only
- ✅ Never commit `.env` to git
- ✅ Use different keys for development and production
- ❌ Never hardcode API keys in source code
- ❌ Never share API keys publicly

### File Uploads
- ✅ Only upload trusted PDFs
- ✅ Scan PDFs with antivirus before upload
- ❌ Don't upload sensitive patient data without proper authorization
- ❌ Don't process files from untrusted sources

### Deployment
- ✅ Keep dependencies updated
- ✅ Use HTTPS in production
- ✅ Implement authentication if exposing publicly
- ✅ Monitor logs for suspicious activity
- ❌ Don't expose app directly to internet without security measures

### Data Privacy
- ✅ Review ChromaDB data retention policies
- ✅ Clear sensitive data regularly
- ✅ Implement proper access controls
- ❌ Don't store PHI without HIPAA compliance measures

## Known Security Considerations

### Current Implementation
- This application is designed for **research and educational purposes**
- Not HIPAA compliant in default configuration
- API keys required for LLM services
- Local data storage in ChromaDB

### Recommendations for Production Use
1. Implement user authentication
2. Add rate limiting
3. Enable HTTPS/TLS
4. Implement audit logging
5. Add input validation and sanitization
6. Regular security audits
7. Compliance review (HIPAA, GDPR, etc.)

## Security Updates

Security updates will be released as soon as possible and announced via:
- GitHub Security Advisories
- Release notes
- README updates

## Contact

For security issues: Use GitHub Security Advisories (preferred)
For general questions: Open a regular GitHub issue

---

**Thank you for helping keep VAID JI secure!** 🔒
