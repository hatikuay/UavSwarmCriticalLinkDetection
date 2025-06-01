# Contributing Guidelines

**Prepared by Dr. Ercan Erkalkan**  
Email: ercan.erkalkan@marmara.edu.tr

Thank you for your interest in contributing to the UAV Swarm Connectivity Simulator! To keep the codebase consistent and maintain high quality, please follow these steps:

1. **Fork the Repository**  
   - Create your branch:  
     ```bash
     git checkout -b feature/your_feature_name
     ```

2. **Coding Standards**  
   - Use clear, self-documenting variable names.  
   - Provide docstrings for all classes and functions.  
   - If you modify existing functions, update their docstrings to reflect new behavior.

3. **Testing**  
   - Write unit tests for new functionality.  
   - Use `pytest` to run tests:  
     ```bash
     pytest --maxfail=1 --disable-warnings -q
     ```
   - Ensure all existing tests still pass.

4. **Linting**  
   - Run a linter (e.g. `flake8`) before you commit:  
     ```bash
     flake8 .
     ```
   - Fix any PEP-8 violations or other style issues.

5. **Pull Request (PR)**  
   - Submit a PR to the `main` branch.  
   - In your PR description:
     - Summarize the change.  
     - Reference any open issue (e.g., “Fixes #12”).  
     - Include before/after screenshots if GUI code was affected.  
   - Wait for at least one approval from the maintainers.  

6. **Code of Conduct**  
   - Be respectful in all communication.  
   - Follow the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

Thank you again for helping improve this project!
