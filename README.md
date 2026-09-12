# hepdataset
Software for making datasets ready for ML related use for HEP studies

## Installation

### Installing via pip

The easiest way to install `hepdataset` is through pip:

```bash
pip install hepdataset
```

This will install the latest stable version of the package along with all required dependencies.

## Usage

### Using in Python Scripts

You can import and use `hepdataset` directly in your Python scripts:

```python
from hepdataset import DatasetProcessor

# Initialize the dataset processor
processor = DatasetProcessor()

# Load your HEP data
dataset = processor.load_data("path/to/your/data")

# Prepare dataset for ML
prepared_data = processor.prepare_for_ml(dataset)

# Save the processed dataset
processor.save(prepared_data, "output/path")
```

For more detailed examples and API documentation, please refer to the [documentation](docs/) or check out the [examples](examples/) directory.

### Using from the Command Line

`hepdataset` also provides a command-line interface for quick data processing without writing Python code:

```bash
# View available commands and options
hepdataset --help

# Process a dataset
hepdataset process --input path/to/data --output output/path

# Validate your dataset
hepdataset validate --input path/to/data
```

Run `hepdataset <command> --help` for more details on any specific command.

## Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding new features, improving documentation, or suggesting enhancements, your help is appreciated.

### How to Contribute

1. **Fork the repository** - Create your own fork of the project
2. **Create a branch** - Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** - Implement your feature or fix
4. **Commit with clear messages** - Write descriptive commit messages
5. **Push to your branch** - Push your changes to your fork
6. **Submit a pull request** - Open a PR with a clear description of your changes

### Guidelines

- Please follow the existing code style and conventions
- Add tests for new features or bug fixes
- Update documentation as needed
- Ensure all tests pass before submitting a PR

We look forward to your contributions!

## Contact

For questions or feedback, please open an issue on the [GitHub repository](https://github.com/ammelsayed/hepdataset).
