FROM python:3.7.9

#  cron installl
RUN apt-get update && apt-get install -y cron build-essential

# work directory
WORKDIR /app
COPY run.py model.py data.py utils.py telecom.csv telecom_test.csv requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

RUN python model.py

# cron job with exec rights
RUN echo "0 8 * * 2 /usr/local/bin/python /app/run.py >> /var/log/cron.log 2>&1" > /etc/cron.d/my-cron-job
RUN chmod 0644 /etc/cron.d/my-cron-job
RUN crontab /etc/cron.d/my-cron-job

# log file
RUN touch /var/log/cron.log

# Run the cron service in the foreground
CMD ["cron", "-f"]