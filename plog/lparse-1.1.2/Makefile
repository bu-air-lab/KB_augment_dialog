INSTALLATION_PATH=/usr/local/bin

lparse :
	cd src ; make ; cd ..

check :
	cd test; make; cd ..

clean :
	cd src; make clean; cd ..
	cd test; make clean; cd ..

install : lparse
	cp src/lparse $(INSTALLATION_PATH)/
