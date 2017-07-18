/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *     Student: Cédric Bodet
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

std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 100;

  // Initialize weights to 1
  weights = std::vector<double>((unsigned long) num_particles, 1.);


  // Initialize distributions
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  // Randomly initialize all particles first position
  particles = std::vector<Particle>((unsigned long) num_particles);
  for(int i=0; i<num_particles; i++) {
    Particle particle;

    particle.x=dist_x(gen);
    particle.y=dist_x(gen);
    particle.theta=dist_theta(gen);

    particles[i]=particle;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Initialize distributions
  std::normal_distribution<double> dist_x(std_pos[0]);
  std::normal_distribution<double> dist_y(std_pos[1]);
  std::normal_distribution<double> dist_theta(std_pos[2]);

  for(int i=0; i<num_particles; i++) {
    Particle& p = particles[i];

    // Predict the position of the particle, avoiding divisions by zero
    if(yaw_rate>0.0001) {
      p.x += velocity/yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
      p.y += velocity/yaw_rate * (cos(p.theta) - cos(p.theta+yaw_rate*delta_t));
      p.theta += delta_t * yaw_rate;
    } else {
      p.x += velocity*delta_t*cos(p.theta);
      p.y += velocity*delta_t*sin(p.theta);
    }

    // Add random gaussian noise
    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

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
}

void ParticleFilter::resample() {
  // A discrete distribution that produces random integers, where the probability of each individual integer
  // is given by the provided weights.
  std::discrete_distribution<> sampling(weights.begin(), weights.end());

  // Weighted resampling with replacements
  std::vector<Particle> old_particles(particles);
  for(int i = 0; i<num_particles; i++) {
    particles[i]=old_particles[sampling(gen)];
  }
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
