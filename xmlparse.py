import xml
import os
import re

from xml.etree import ElementTree
file_name="books.xml"

#full_file=os.path.join('data',file_name)
full_file=os.path.abspath(os.path.join('',file_name))
print (full_file)


dom=ElementTree.parse(full_file)

#with the above class , we parsed a file into a document object model .where we could run xpath expressions
books=dom.findall('book')

for t in books:
    titles=t.find('title').text
    authors=t.find('author').text
    genres=t.find('genre').text

    print ('* {} {} {} {} {}'.format(titles,'written by' ,authors, 'of genre -', genres) )
