from google_drive_downloader import GoogleDriveDownloader as gdd
import os

GOOGLE_DRIVE_CHECKPOINT_ID = "1GimBuE7QPwId64TOpJfh1Jhs8j0WPGj3"
GOOGLE_DRIVE_CONFIG_ID = "1nwgQUu-w8Tf2uuhiIcfgHOpCt8euZH6_"

DIR = "default_test_model"


if not os.path.exists(DIR):
    os.mkdir(DIR)

gdd.download_file_from_google_drive(file_id=GOOGLE_DRIVE_CHECKPOINT_ID,
                                    dest_path=os.path.join(DIR, "checkpoint.pth"),
                                    unzip=False)
gdd.download_file_from_google_drive(file_id=GOOGLE_DRIVE_CONFIG_ID,
                                    dest_path=os.path.join(DIR, "config.json"),
                                    unzip=False)
