/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang, Maxim Kulesh
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Initialize Particle Filter:
	// Number of particles: num_particles
	// Particles are set to position x, y provided by GPS with standard deviation std[]
	if (is_initialized) return;
	num_particles = 100;
	default_random_engine gen;
	normal_distribution<double> pos_x(x, std[0]);
	normal_distribution<double> pos_y(y, std[1]);
	normal_distribution<double> pos_yaw(theta, std[2]);
	for (int i = 0; i < num_particles; ++i)
	{
		Particle p = {.id = i, .x = pos_x(gen), .y = pos_y(gen), .theta = pos_yaw(gen), .weight = 1.0};
		particles.push_back(p);
		weights.push_back(1.0);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	default_random_engine gen;
	normal_distribution<double> std_x(0.0, std_pos[0]);
	normal_distribution<double> std_y(0.0, std_pos[1]);
	normal_distribution<double> std_yaw(0.0, std_pos[2]);
	double eps = 1e-4;
	vector<Particle>::iterator p;
	if (fabs(yaw_rate) > eps)
	{
		for(p = particles.begin(); p != particles.end(); ++p)
		{
			double curv = velocity / yaw_rate;
			double new_theta = p->theta + yaw_rate * delta_t;
			p->x += curv * (sin(new_theta) - sin(p->theta)) + std_x(gen);
			p->y += curv * (cos(p->theta) - cos(new_theta)) + std_y(gen);
			p->theta = normAngle(new_theta + std_yaw(gen));
		}
		return;
	}
	for(p = particles.begin(); p != particles.end(); ++p)
	{
		p->x += velocity * delta_t * cos(p->theta) + std_x(gen);
		p->y += velocity * delta_t * sin(p->theta) + std_y(gen);	
		p->theta = normAngle(p->theta + yaw_rate * delta_t + std_yaw(gen));
	}
}

double ParticleFilter::errorAssociation(double std[], std::vector<LandmarkObs> landmarks, std::vector<LandmarkObs> observations) {
	double stdx_inv = 1 / (2 * std[0]);
	double stdy_inv = 1 / (2 * std[1]);
	double error = 0.0;

	vector<LandmarkObs>::iterator obs;
	vector<LandmarkObs>::iterator lmark;
	for(obs = observations.begin(); obs != observations.end(); ++obs)
	{
		double min_dist = std::numeric_limits<double>::infinity();
		for(lmark = landmarks.begin(); lmark != landmarks.end(); ++lmark)
		{
			double d_x = lmark->x - obs->x;
			double d_y = lmark->y - obs->y;	
			double dist = d_x * d_x * stdx_inv + d_y * d_y * stdy_inv;
			if (dist < min_dist)
			{
				min_dist = dist;
			}
		}
		// instead of taking exponent of each error and then multiplying, it is much more computationally efficient to sum the errors and take a single exponent	
		error += min_dist;	
	}
	return error;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	
	vector<Particle>::iterator p;
	vector<Map::single_landmark_s>::const_iterator lm;
	vector<LandmarkObs>::const_iterator obs;
	int i = 0;
	for (p = particles.begin(); p != particles.end(); ++p)
	{
		vector<LandmarkObs> local_lmarks;
		for (lm = map_landmarks.landmark_list.begin(); lm != map_landmarks.landmark_list.end(); ++lm)
		{
			if( dist(p->x, p->y, (double) lm->x_f, (double) lm->y_f) <= sensor_range)
				local_lmarks.push_back((LandmarkObs){lm->id_i, (double) lm->x_f, (double) lm->y_f});
		}
		vector<LandmarkObs> global_obs;	
		for (obs = observations.begin(); obs != observations.end(); ++obs)
			global_obs.push_back((LandmarkObs){obs->id, obs->x * cos(p->theta) - obs->y * sin(p->theta) + p->x, obs->x * sin(p->theta) + obs->y * cos(p->theta) + p->y}); 
		double error = errorAssociation(std_landmark, local_lmarks, global_obs);
		// Note: no need to normalize with (2*pi*std_x*std_y), as global normalization will be done later anyway
		weights[i++] = exp(-error);
	}
	// normalize weights
	double sum_weights_inv = 1/accumulate(weights.begin(), weights.end(), 0.0);
	for (int i = 0; i < num_particles; ++i)
	{
		weights[i] *= sum_weights_inv;
		particles[i].weight = weights[i];
	}
}

void ParticleFilter::resample() {
	default_random_engine gen;
	discrete_distribution<> dist(weights.begin(), weights.end());
	vector<Particle> resampled;
	for(int i = 0; i < num_particles; ++i)
	{
		resampled.push_back(particles[dist(gen)]);
	}
	particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    	particle.associations= associations;
  	particle.sense_x = sense_x;
    	particle.sense_y = sense_y;
	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    	string s = ss.str();
    	s = s.substr(0, s.length()-1);  // get rid of the trailing space
    	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    	string s = ss.str();
    	s = s.substr(0, s.length()-1);  // get rid of the trailing space
    	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    	string s = ss.str();
    	s = s.substr(0, s.length()-1);  // get rid of the trailing space
    	return s;
}
