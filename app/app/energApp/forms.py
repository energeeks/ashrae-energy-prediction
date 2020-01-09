from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, validators, SubmitField
from wtforms.validators import ValidationError, DataRequired, EqualTo, Length


class SignupForm(FlaskForm):
    name = StringField('Name',
                       validators=[DataRequired(message=('Please enter a nickname.'))])
    password = PasswordField('Password',
                             validators=[DataRequired(message='Please enter a password.'),
                                         Length(min=3, message=('Please select a stronger password.')),
                                         EqualTo('confirm', message='Passwords must match')])
    confirm = PasswordField('Confirm Your Password', )
    submit = SubmitField('Register')


class LoginForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired('Please enter a valid name.')])
    password = PasswordField('Password', validators=[DataRequired('Please enter your password.')])
    submit = SubmitField('Log In')
