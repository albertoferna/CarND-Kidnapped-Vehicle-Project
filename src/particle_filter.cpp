/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
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
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 100;
	is_initialized = true;
    default_random_engine gen;
	normal_distribution<double> gaussian_x(x, std[0]);
	normal_distribution<double> gaussian_y(y, std[1]);
	normal_distribution<double> gaussian_theta(theta, std[2]);
	for (int i=0; i<num_particles; i++)
	{
        Particle a_particle;
        a_particle.id = i;
        a_particle.x = gaussian_x(gen);
        a_particle.y = gaussian_y(gen);
        a_particle.theta = gaussian_theta(gen);
        a_particle.weight = 1.0;
        particles.push_back(a_particle);
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    // temporal variables
    double vel_yaw;
    double yawxdt;
    // check if driving in a straight line
    bool straight = false;
    if (abs(yaw_rate) < 0.0001) {
        straight = true;
    }
    if (!straight) {
        vel_yaw = velocity / yaw_rate;
        yawxdt = yaw_rate * delta_t;
    }

    default_random_engine gen;
    normal_distribution<double> gaussian_x(0.0, std_pos[0]);
    normal_distribution<double> gaussian_y(0.0, std_pos[1]);
    normal_distribution<double> gaussian_theta(0.0, std_pos[2]);

    for (int i = 0; i < num_particles; i++) {
        double theta = particles[i].theta;
        if (!straight) {
            particles[i].x += vel_yaw * (sin(theta + yawxdt) - sin(theta));
            particles[i].y += vel_yaw * (-cos(theta + yawxdt) + cos(theta));
            particles[i].theta += yawxdt;
        } else {
            particles[i].x += velocity * cos(theta) * delta_t;
            particles[i].y += velocity * sin(theta) * delta_t;
        }

        // add gaussian noise to prediction
        particles[i].x += gaussian_x(gen);
        particles[i].y += gaussian_y(gen);
        particles[i].theta += gaussian_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (auto &observ : observations) {
        double min_dist = 1e15; // initialize with a very big number
        for (unsigned int i = 0; i<predicted.size(); i++) {
            LandmarkObs predict = predicted[i];
            if (dist(observ.x, observ.y, predict.x, predict.y) < min_dist) {
                observ.id = predict.id;
                min_dist = dist(observ.x, observ.y, predict.x, predict.y);
            }
        }
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    for (auto &particle : particles) {
        std::vector<LandmarkObs> potential;
        // given the position of the particle filter landmarks that would be too far away
        for (auto &landmark : map_landmarks.landmark_list)
        if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) <= sensor_range * 1.1) {
            LandmarkObs landmrk;
            landmrk.id = landmark.id_i;
            landmrk.x = landmark.x_f;
            landmrk.y = landmark.y_f;
            potential.push_back(landmrk);
        }
        // Need to create a vector to keep transformed observations as they are
        // function of the assumed particle that sense them
        std::vector<LandmarkObs> trasn_obs = observations;
        for (auto &observation : trasn_obs) {
            double x_mod = particle.x + cos(particle.theta) * observation.x - sin(particle.theta) * observation.y;
            double y_mod = particle.y + sin(particle.theta) * observation.x + cos(particle.theta) * observation.y;
            observation.x = x_mod;
            observation.y = y_mod;
        }
        dataAssociation(potential, trasn_obs);
        // calculate weight of particle
        double weight = 1.0;
        double sig_x = std_landmark[0];
        double sig_y = std_landmark[1];
        // calculate normalization term
        double gauss_norm= (1/(2 * M_PI * sig_x * sig_y));

        for (auto &observation : trasn_obs) {
            int land_id = observation.id;
            // carefull here, landmark id is not the list index of that landmark, ids start at 1
            double mu_x = map_landmarks.landmark_list[land_id-1].x_f;
            double mu_y = map_landmarks.landmark_list[land_id-1].y_f;
            double dx = observation.x - mu_x;
            double dy = observation.y - mu_y;
            // calculate exponent
            // exponent = ((x_obs - mu_x)**2)/(2 * sig_x**2) + ((y_obs - mu_y)**2)/(2 * sig_y**2);
            double exponent = (dx*dx)/(2*sig_x*sig_x) + (dy*dy)/(2*sig_y*sig_y);
            // calculate weight using normalization terms and exponent
            weight *= gauss_norm * exp(-exponent);
        }
        particle.weight = weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::vector<double> weights;
    // get weights of current particles
    for (auto &particle : particles) {
        weights.push_back(particle.weight);
    }

    default_random_engine gen;
    // create a discrete distribution defined by the weights vector
    // weights not necesarly formal probabilities, they are divided by the total in the generator
    std::discrete_distribution<> distribution(weights.begin(), weights.end());

    std::vector<Particle> new_particles;
    for (unsigned i = 0; i < particles.size(); i++)
        // distribution(gen) generates an integer index according to its respective weight in weights
        new_particles.push_back(particles[distribution(gen)]);

    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

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
