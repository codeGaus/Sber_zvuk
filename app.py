from flask import Flask
import aioboto3
import boto3
from flask import request
import requests
from ml import create_video
from static.constants import ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

session = boto3.session.Session()

s3 = session.client(
    service_name='s3',
    endpoint_url=ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    use_ssl=False,
    verify=False,
)


app = Flask(__name__)


@app.route('/')
def home():
    return 'Hello World!'


@app.route('/recognize')
def recognize():
    prefix = request.args.get('prefix', default='default_name', type=str)
    source = request.args.get('source', default='https://hackaton.sber-zvuk.com/hackathon_part_1.mp4', type=str)

    print("The file download has started...Wait")
    with open(f"{prefix}.mp4", 'wb') as file:
        video_content = requests.get(source)
        file.write(video_content.content)
    print("The file has been downloaded successfully.")

    create_video(source, prefix)

    print("Uploading content to amazon bucket...")
    s3.upload_file(f'{prefix}_result.mp4', 'hackathon-ecs-49', f'{prefix}_result.mp4')
    print(s3.list_objects(Bucket='hackathon-ecs-49'))
    print("Content upload to amazon bucket completed successfully.")

    return {
        "code": "200 OK",
        "message ": "Content processing completed."
    }


if __name__ == '__main__':
    app.run()
