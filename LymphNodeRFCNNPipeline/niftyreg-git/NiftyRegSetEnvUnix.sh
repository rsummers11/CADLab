#!/bin/sh

# $0 Script path
# $1 Package path
# $2 Target location
# $3 Target Volume


TO_ADD_BASH="\n"
TO_ADD_CSH="\n"
TO_ADD_BASH="${TO_ADD_BASH}\n# NIFTYREG install_path=$2"
TO_ADD_CSH="${TO_ADD_CSH}\n# NIFTYREG install_path=$2"
TO_ADD_BASH="${TO_ADD_BASH}\nexport PATH=\${PATH}:$2/bin"
TO_ADD_CSH="${TO_ADD_CSH}\nsetenv PATH \${PATH}:$2/bin"
TO_ADD_BASH="${TO_ADD_BASH}\nexport LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:$2/lib"
TO_ADD_CSH="${TO_ADD_CSH}\nsetenv LD_LIBRARY_PATH \${LD_LIBRARY_PATH}:$2/lib"
if [ "${OS}" == "Darwin" ]; then
        TO_ADD_BASH="${TO_ADD_BASH}\nexport DYLD_LIBRARY_PATH=\${DYLD_LIBRARY_PATH}:$2/lib"
        TO_ADD_CSH="${TO_ADD_CSH}\nsetenv DYLD_LIBRARY_PATH \${DYLD_LIBRARY_PATH}:$2/lib"
fi
if [ -f ~/.profile ]
then
	temp=`cat ~/.profile | grep "# NIFTYREG install_path=$2"`
	if [ "${temp}" == "" ]
	then
		echo "${TO_ADD_BASH}" >> ~/.profile
	fi
fi
if [ -f ~/.bashrc ]
then
	temp=`cat ~/.bashrc | grep "# NIFTYREG install_path=$2"`
	if [ "${temp}" == "" ]
	then
		echo "${TO_ADD_BASH}" >> ~/.bashrc
	fi
fi
if [ -f ~/.cshrc ]
then
	temp=`cat ~/.cshrc | grep "# NIFTYREG install_path=$2"`
	if [ "${temp}" == "" ]
	then
		echo "${TO_ADD_BASH}" >> ~/.cshrc
	fi
fi

