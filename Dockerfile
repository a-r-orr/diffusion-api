# 1 Docker Image
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# 2 Working Directory and non-root user
WORKDIR /usr/src/app
RUN useradd --create-home appuser
USER appuser
# Add the user's local bin directory to the PATH environment variable
ENV PATH="/home/appuser/.local/bin:${PATH}"

# 3 Install dependencies
COPY --chown=appuser:appuser requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4 Copy application source code
COPY --chown=appuser:appuser src/ ./src

# 5 Copy startup file
COPY --chown=appuser:appuser helper_script.sh ./
RUN chmod +x helper_script.sh

# 6 Expose PORT 8080 for Gunicorn
EXPOSE 8080

# 7 Command to run application
CMD ["./helper_script.sh"]