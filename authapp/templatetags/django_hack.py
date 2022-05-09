#from google.appengine.ext import webapp

#register = webapp.template.create_template_register()

from django import template

register = template.Library()
def hash(h,key):
    if key in h:
        return h[key]
    else:
        return None

register.filter(hash)