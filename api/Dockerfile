FROM python:3.8.13 as base

# it should be collected by env file, TODO
ARG UID=501
ARG GID=570

# Add a non-root user
RUN addgroup --gid $GID app

# adduser command from https://linux.die.net/man/8/adduser
# useradd command from https://linux.die.net/man/8/useradd
# usermod command from https://linux.die.net/man/8/usermod
RUN adduser --disabled-login --geco '' --uid $UID --gid $GID app

# NEW
RUN mkdir src

# Append /home/app/.local/bin/ to PATH variable because
# gunicorn is installed there.
ENV PYTHONPATH=$PYTHONPATH:/src/
ENV PATH=$PATH:/home/app/.local/bin/

COPY ./ /src

WORKDIR /src

# (currently executing command as a root user)

# add current user and group inside of the src directory recursively
# user app and group app 
RUN chown -R app:app /src/
# From Alan Bauzá
# add read and write permissions for users
# give only read permissions for group
# recursively in src directory
RUN chmod u+rw,g=r -R /src/

# finally use app account 
USER app

RUN pip install --upgrade pip && pip install -r requirements.txt

FROM base as test
RUN ["pytest", "-v", "/src/tests"]

FROM base as build
#ENTRYPOINT ["gunicorn", "--workers=8", "--bind", "0.0.0.0:5000", "app:app"]
ENTRYPOINT ["gunicorn", "--workers=8", "--bind", "0.0.0.0:5000", "--timeout", "300", "app:app"]