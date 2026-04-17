from __future__ import annotations

import uvicorn

import server_config


def main() -> None:
    uvicorn.run(
        server_config.APP_IMPORT,
        host=server_config.HOST,
        port=server_config.PORT,
        reload=server_config.RELOAD,
        log_level=server_config.LOG_LEVEL,
    )


if __name__ == "__main__":
    main()
