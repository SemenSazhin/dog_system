# Настройка синхронизации с Colab (один раз)

Colab хранит ноутбуки на Google Drive, поэтому синхронизация идёт через Drive API от имени вашего аккаунта.

1. [Google Cloud Console](https://console.cloud.google.com/) → создать проект → **APIs & Services → Library → Google Drive API → Enable**.
2. **OAuth consent screen**: тип External, добавить свой Gmail в Test users. Затем **Publish app** (иначе refresh-токен протухает через 7 дней).
3. **Credentials → Create credentials → OAuth client ID → Desktop app**, скачать JSON (`client_secret.json`).
4. Локально:
   ```bash
   pip install google-auth-oauthlib
   python scripts/get_refresh_token.py client_secret.json
   ```
   Войти в Google, разрешить доступ. Скрипт напечатает три значения.
5. GitHub → репозиторий → **Settings → Secrets and variables → Actions** → добавить секреты
   `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REFRESH_TOKEN`.
6. Пуш в `master` (или **Actions → Colab sync → Run workflow**) — ноутбуки загрузятся на Drive в папку с именем репозитория, появится `README.md` со ссылками.

## Как работает

- Пуш с изменёнными `.ipynb` → новые создаются на Drive, существующие обновляются (ссылка на Colab не меняется).
- Раз в 30 минут (и вручную) проверяется Drive: если ноутбук изменили в Colab (**File → Save**), он скачивается и коммитится в репо.
- Если файл изменён с обеих сторон: при пуше побеждает репозиторий (версия из Colab остаётся в истории версий файла на Drive), по расписанию — пропуск с предупреждением.
- Состояние хранится в `.colab-sync.json`, `README.md` генерируется автоматически.
- Приложение видит только файлы, созданные самим скриптом (scope `drive.file`).
