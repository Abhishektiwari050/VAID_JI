# Contributing to VAID JI

Thank you for your interest in contributing to VAID JI! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- A clear description of the problem
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)

### Suggesting Features

We welcome feature suggestions! Please:
- Check if the feature has already been requested
- Provide a clear use case
- Explain how it benefits users

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Set up your environment**:
   ```bash
   python -m venv medical_assistant_env
   source medical_assistant_env/bin/activate  # On Windows: medical_assistant_env\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

3. **Make your changes**:
   - Follow PEP 8 style guidelines
   - Add docstrings to functions and classes
   - Include type hints where appropriate
   - Update documentation if needed

4. **Test your changes**:
   - Ensure existing functionality still works
   - Add tests for new features
   - Run `python -m pytest` if tests are available

5. **Commit your changes**:
   - Use clear, descriptive commit messages
   - Reference issues in commits (e.g., "Fix #123")

6. **Submit a Pull Request**:
   - Provide a clear description of changes
   - Link related issues
   - Ensure CI checks pass

## 📝 Code Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Keep functions focused and concise
- Add comments for complex logic
- Use type hints for function signatures

## 🧪 Testing

- Write tests for new features
- Ensure tests pass before submitting PR
- Include both positive and negative test cases

## 📚 Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update inline comments as needed
- Keep documentation clear and concise

## 🔒 Security

- Never commit API keys or credentials
- Use environment variables for sensitive data
- Report security issues privately to the maintainers

## 💬 Communication

- Be respectful and constructive
- Ask questions if you're unsure
- Help others in the community

## 📜 Code of Conduct

- Be welcoming and inclusive
- Respect differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community

Thank you for contributing to VAID JI! 🙏
