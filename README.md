
# Compilation

This software is written in C++ and requires

- GNU g++ Version 6.x or greater
- GNU fortran Version 6.x or greater
- GNU Make version 4.x or greater
- GNU Scientific Library (GSL) version 1 or 2
- OpenMPI version 1.10

In the root of the directory:
```
> make 
```

## Snow and ice retrieval

Inside of the `snow_ice` folder there are a few models that can be used. In order to run them they first need to be compiled calling:
```
> make
```

And then run the inversion using:
```
> python InversionBinnedParallel.py
```


