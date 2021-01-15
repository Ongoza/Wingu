from os.path import isfile
# from envparse import env

#if isfile('local.env'):
#    env.read_envfile('local.env')

DEBUG = False # env.bool('DEBUG', default=False)

SITE_HOST = "http://127.0.0.1" #env.str('HOST')
SITE_PORT = 8080 # env.int('PORT')
SECRET_KEY = "Test_session_key" #env.str('SECRET_KEY')
# MONGO_HOST = env.str('MONGO_HOST')
DB_PATH = "db.sqlite3" # env.str('DB_PATH')
DB_NAME = "CACS"
STATS_COLLECTION = 'stats'
INTERSECTION_COLLECTION = 'intersetions'
# MESSAGE_COLLECTION = 'posts'
USER_COLLECTION = 'users'
# generate password hash
ADMIN_USER = 'admin'
# echo -n password_string | md5sum
ADMIN_PASSWORD = '09613b137a2e9be2ae93588c5970ab53'
