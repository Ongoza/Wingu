from os.path import isfile
from envparse import env

if isfile('local.env'):
    env.read_envfile('local.env')

DEBUG = env.bool('DEBUG', default=False)

SITE_HOST = env.str('HOST')
SITE_PORT = env.int('PORT')
SECRET_KEY = env.str('SECRET_KEY')
# MONGO_HOST = env.str('MONGO_HOST')
DB_PATH = env.str('DB_PATH')

# MESSAGE_COLLECTION = 'posts'
USER_COLLECTION = 'users'
# generate password hash
ADMIN_USER = 'admin'
# echo -n password_string | md5sum
ADMIN_PASSWORD = '09613b137a2e9be2ae93588c5970ab53'
