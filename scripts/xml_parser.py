import xml.etree.ElementTree as ET

tree = ET.parse("factura.xml")
root = tree.getroot()

print(root.tag)
