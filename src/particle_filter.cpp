/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *     Student: Cédric Bodet
 */

#include <random>
#include <iostream>
#include <sstream>

#include "particle_filter.h"

using namespace std;

std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 500;

  // Initialize weights to 1
  weights = std::vector<double>((unsigned long) num_particles, 1.);


  // Initialize distributions
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  // Randomly initialize all particles first position
  particles = std::vector<Particle>((unsigned long) num_particles);
  for (int i = 0; i < num_particles; i++) {
    Particle particle;

    particle.x = dist_x(gen);
    particle.y = dist_x(gen);
    particle.theta = dist_theta(gen);

    particles[i] = particle;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Initialize distributions
  std::normal_distribution<double> dist_x(std_pos[0]);
  std::normal_distribution<double> dist_y(std_pos[1]);
  std::normal_distribution<double> dist_theta(std_pos[2]);

  for (Particle &p : particles) {
    // Predict the position of the particle, avoiding divisions by zero
    if (yaw_rate > 0.0001) {
      p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
      p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
      p.theta += delta_t * yaw_rate;
    } else {
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
    }

    // Add random gaussian noise
    p.x += dist_x(gen);
    p.y += dist_y(gen);
    p.theta += dist_theta(gen);
  }
}

/*
 * Computes the Euclidean squared distance between two 2D points.
 * @param (x1,y1) x and y coordinates of first point
 * @param (x2,y2) x and y coordinates of second point
 * @output Euclidean squared distance between two 2D points
 */
inline double dist2(double x1, double y1, double x2, double y2) {
  return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
}

inline double dist2(const LandmarkObs &a, const Map::single_landmark_s &b) {
  return dist2(a.x, a.y, b.x_f, b.y_f);
}

inline double dist2(const Particle &a, const Map::single_landmark_s &b) {
  return dist2(a.x, a.y, b.x_f, b.y_f);
}

const Map::single_landmark_s &
ParticleFilter::findClosestLandmark(const std::vector<Map::single_landmark_s> &landmarks, const LandmarkObs &map_obs) {
  double min_dist2 = dist2(map_obs, landmarks[0]);
  int closest_i = 0;

  for (int i = 1; i < landmarks.size(); i++) {
    double current_dist2 = dist2(map_obs, landmarks[i]);
    if (current_dist2 < min_dist2) {
      min_dist2 = current_dist2;
      closest_i = i;
    }
  }
  return landmarks[closest_i];
}

inline double gaussian_exp(double x, double x1, double sigma) {
  return exp(-(x - x1) * (x - x1) / (2 * sigma * sigma));
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {

  const double sigma_x = std_landmark[0];
  const double sigma_y = std_landmark[1];
  const double prefactor = 1. / (2. * M_PI * sigma_x * sigma_y);

  weights.clear();

  for (Particle &p : particles) {

    // Filter landmarks in range (only consider landmarks in range of the mean particle, with some buffer)
    std::vector<Map::single_landmark_s> landmarks_in_range;
    for (const Map::single_landmark_s &landmark : map_landmarks.landmark_list) {
      if (dist2(p, landmark) < 1.2 * sensor_range * sensor_range) {
        landmarks_in_range.push_back(landmark);
      }
    }

    std::vector<LandmarkObs> map_observations;
    double final_weight = 1.;

    // For each observation
    for (LandmarkObs &obs : observations) {
      // Transform the observation to map coordinates
      LandmarkObs map_obs;
      map_obs.x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
      map_obs.y = obs.x * sin(p.theta) + obs.y * cos(p.theta) + p.y;

      // Find nearest neighbour and assign observation
      const Map::single_landmark_s &closest = findClosestLandmark(landmarks_in_range, map_obs);
      map_obs.id = closest.id_i;

      // Calculate weight
      double weight = prefactor
                      * gaussian_exp(map_obs.x, closest.x_f, sigma_x)
                      * gaussian_exp(map_obs.y, closest.y_f, sigma_y);
      final_weight *= weight;

      map_observations.push_back(map_obs);
    }

    // Store landmark associations
    p = SetAssociations(p, map_observations);

    // Update the weights
    p.weight = final_weight;
    weights.push_back(final_weight);
  }
}

void ParticleFilter::resample() {
  // A discrete distribution that produces random integers, where the probability of each individual integer
  // is given by the provided weights.
  std::discrete_distribution<> sampling(weights.begin(), weights.end());

  // Weighted resampling with replacements
  std::vector<Particle> old_particles(particles);
  for (int i = 0; i < num_particles; i++) {
    particles[i] = old_particles[sampling(gen)];
  }
}

Particle ParticleFilter::SetAssociations(Particle particle, const std::vector<LandmarkObs> &map_observations) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // map_observations: the list of associations with landmark id, x and y converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  for (const LandmarkObs &obs : map_observations) {
    particle.associations.push_back(obs.id);
    particle.sense_x.push_back(obs.x);
    particle.sense_y.push_back(obs.y);
  }

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
