# Python Code Style Rules

## Python Version
- Target Python >= 3.11

## Formatting
- Use `black` compatible formatting
- Max line length: 88

## Typing
- Use modern Python typing
- Avoid `typing.List`, `typing.Dict`
- Prefer `list[str]`, `dict[str, int]`

## Imports
Order imports:

1. standard library
2. third-party
3. local modules

Example:

```python
import os
from pathlib import Path

from pytube import YouTube

from .utils import sanitize_filename
```

## Strings
- Prefer single quotes
- Use f-strings for formatting
- Use lazy formatting for logging

Example:

```python
logger.info('Downloading %s', title)
```

## Logging
- Use `logging`, never `print`
- Default level: INFO

## Error Handling
- Catch specific exceptions
- Never use bare `except`

Good:

```python
except HTTPError as error:
```

Bad:

```python
except:
```

## Functions
- Maximum ~40 lines
- One responsibility per function

## Filenames
Always sanitize user input:

```python
sanitize_filename(name)
```

## Path Handling
Use `pathlib.Path`, never raw string paths.

## Environment Variables
Load from `.env` using:

```python
from dotenv import load_dotenv
```

## CLI Safety
Scripts must validate required environment variables before execution.

## Networking
Network calls must always be wrapped in exception handling.
