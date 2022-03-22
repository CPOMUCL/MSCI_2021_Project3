#ifndef snowicesynthetic_hpp
#define snowicesynthetic_hpp

double tidesynthetic_horizontal_cosine1(double nx, double ny);

double tidesynthetic_vertical_cosine1(double nx, double ny);

double tidesynthetic_horizontal_cosine2(double nx, double ny);

double tidesynthetic_vertical_cosine2(double nx, double ny);

double tidesynthetic_horizontal_cosine3(double nx, double ny);

double tidesynthetic_vertical_cosine3(double nx, double ny);

double tidesynthetic_tas_sea(double nx, double ny);

double tidesynthetic_tas_land(double nx, double ny);

double synthetic_gaussian_snow(double nx, double ny);

double synthetic_gaussian_ice(double nx, double ny);

#endif // tidesynthetic_hpp
