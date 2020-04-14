include ../make.mk

IFLAGS = -I$(CPPMPC_HEADER) \
	     -I$(QPOASES_HEADER)

# OBJ += InvertedPendulum.o
OBJ += Quadrotor.o

# OUT_DYNAMIC += InvertedPendulum_d
OUT_DYNAMIC += Quadrotor_d

all: $(OUT_DYNAMIC)

$(OUT_DYNAMIC): $(OBJ)
	@echo "Creating" $@
	@$(CPP) -o $@ ../model/casadi_src.o $< $(CPPMPC_LINK)

%.o: %.cpp
	@echo "Creating" $@
	@$(CPP) $(CPPFLAGS) -c $(IFLAGS) $<

clean:
	@echo "Cleaning up (examples)"
	@rm -rf *.o *.exe *.txt
