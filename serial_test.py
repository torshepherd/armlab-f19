import serial

with serial.Serial('/dev/ttyACM0, 9600) as s:
	input('Start')
	s.write(b'1')
	input('Press enter to turn off')
	s.write(b'0')