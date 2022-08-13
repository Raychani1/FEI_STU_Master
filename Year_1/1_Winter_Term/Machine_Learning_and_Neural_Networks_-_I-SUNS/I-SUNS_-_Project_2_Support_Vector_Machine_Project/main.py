import os
import platform
from executor.support_vector_machine_project import SupportVectorMachineProject

if __name__ == '__main__':
	# Activate colors in Windows Command Line and Powershell
	if platform.system == 'Windows':
		os.system('color')

	support_vector_machine_project = SupportVectorMachineProject().run()
