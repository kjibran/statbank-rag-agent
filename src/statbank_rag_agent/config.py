from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = ""
    statbank_url: str = "https://api.statbank.dk/v1"
    language: str = "en"


settings = Settings()
