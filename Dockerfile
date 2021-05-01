FROM python:3.6

# Install dependencies
COPY . /code/
WORKDIR /code
RUN pip install -r requirements.txt

# Run examples
RUN python platometer.py examples/folders_to_process.txt
RUN echo "Output below should show 4 .txt files"
RUN ls -l examples/*/platometer*/*.txt
