FROM python:3.12-slim

ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME


RUN pip install poetry==2.2.1
ENV PATH="/home/www/.local/bin:$PATH"

COPY pyproject.toml poetry.lock README.md ./

RUN poetry install

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g $GROUP_ID www && \
    useradd -m -u $USER_ID -g www www

RUN chown -R www:www $APP_HOME
COPY --chown=www:www . $APP_HOME

EXPOSE 80

USER www

CMD ["tail", "-f", "/dev/null"]